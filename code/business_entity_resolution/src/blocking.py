"""
Scalable Blocking & Candidate Generation Module
Uses Country Partitioning + Batched Sparse TF-IDF Cosine Dot-Products.
Ensures high recall while keeping the candidate set compact for leaderboard scoring bonuses.
"""

import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

def run_blocking_for_partition(
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    top_k: int = 15,
    min_similarity: float = 0.12,
    batch_size: int = 2000
) -> Dict[str, List[str]]:
    """
    Runs blocking for a single country partition.
    """
    candidates: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_df["entity_id"]}
    
    if len(s1_df) == 0 or len(targets_df) == 0:
        return candidates
        
    s1_texts = (s1_df["norm_name"].fillna("") + " " + s1_df["norm_name"].fillna("") + " " + s1_df["norm_addr"].fillna("")).tolist()
    target_texts = (targets_df["norm_name"].fillna("") + " " + targets_df["norm_name"].fillna("") + " " + targets_df["norm_addr"].fillna("")).tolist()
    
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        min_df=1,
        max_features=50000,
        sublinear_tf=True
    )
    
    # Fit on targets and transform both sets
    target_matrix = vectorizer.fit_transform(target_texts)
    s1_matrix = vectorizer.transform(s1_texts)
    
    target_ids = targets_df["entity_id"].values
    s1_ids = s1_df["entity_id"].values
    n_s1 = s1_matrix.shape[0]
    
    # Stream in batches to preserve low memory footprint
    for start_idx in range(0, n_s1, batch_size):
        end_idx = min(start_idx + batch_size, n_s1)
        sub_s1_matrix = s1_matrix[start_idx:end_idx]
        
        sim_scores = sub_s1_matrix.dot(target_matrix.T)
        
        for row_i in range(end_idx - start_idx):
            global_s1_idx = start_idx + row_i
            curr_s1_id = s1_ids[global_s1_idx]
            
            row_sims = sim_scores[row_i].toarray().ravel()
            valid_indices = np.where(row_sims >= min_similarity)[0]
            
            if len(valid_indices) == 0:
                continue
                
            if len(valid_indices) > top_k:
                top_part = np.argpartition(row_sims[valid_indices], -top_k)[-top_k:]
                best_indices = valid_indices[top_part]
                best_indices = best_indices[np.argsort(-row_sims[best_indices])]
            else:
                best_indices = valid_indices[np.argsort(-row_sims[valid_indices])]
                
            candidates[curr_s1_id] = [str(target_ids[idx]) for idx in best_indices]
            
    return candidates

def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = 15,
    min_similarity: float = 0.12
) -> Dict[str, List[str]]:
    """
    Main blocking pipeline that partitions by country and emits candidate lists.
    Guarantees every S1 entity is present in the returned mapping.
    """
    logger.info(f"Generating candidates for {len(s1_df):,} Source 1 entities (top_k={top_k})...")
    all_candidates: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_df["entity_id"]}
    
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    countries = s1_df["country"].unique()
    
    total_pairs = 0
    for country in countries:
        s1_c = s1_df[s1_df["country"] == country]
        targets_c = targets_df[targets_df["country"] == country]
        logger.info(f"Partition [{country}]: {len(s1_c):,} S1 vs {len(targets_c):,} Targets")
        
        country_candidates = run_blocking_for_partition(
            s1_c, targets_c, top_k=top_k, min_similarity=min_similarity
        )
        for s1_id, cands in country_candidates.items():
            all_candidates[s1_id] = cands
            total_pairs += len(cands)
            
    avg_cands = total_pairs / max(len(s1_df), 1)
    logger.info(f"Candidate generation complete. Total pairs: {total_pairs:,} | Avg per S1: {avg_cands:.2f}")
    return all_candidates
