"""
Model Training Module
Trains an XGBoost pairwise classifier using Entity-Level splitting (preventing data leakage).
Saves the trained model and F_0.5-optimized threshold.
"""

import os
import json
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit

from .features import FEATURE_COLUMNS, build_feature_matrix
from .threshold import optimize_decision_threshold, compute_macro_f05

logger = logging.getLogger(__name__)

def train_entity_matcher(
    candidates_map: Dict[str, List[str]],
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    ground_truth: Dict[str, List[str]],
    model_output_dir: str = "models",
    random_seed: int = 42
) -> Tuple[XGBClassifier, float]:
    """
    Trains pairwise XGBoost model with GroupShuffleSplit by S1 Entity ID to prevent data leakage.
    """
    os.makedirs(model_output_dir, exist_ok=True)
    logger.info("Extracting features for candidate pairs...")
    
    X_df, pair_ids = build_feature_matrix(candidates_map, s1_df, targets_df)
    
    if len(pair_ids) == 0:
        raise ValueError("No candidate pairs generated. Cannot train model.")
        
    # Construct binary targets
    y = np.zeros(len(pair_ids), dtype=int)
    groups = []
    
    for idx, (s1_id, tgt_id) in enumerate(pair_ids):
        groups.append(s1_id)
        if tgt_id in ground_truth.get(s1_id, []):
            y[idx] = 1
            
    pos_count = int(np.sum(y))
    neg_count = len(y) - pos_count
    logger.info(
        f"Training dataset constructed: {len(y):,} pairs | "
        f"Positives: {pos_count:,} ({pos_count/len(y):.2%}) | Negatives: {neg_count:,}"
    )
    
    # Entity-Level Group Split (80% train, 20% validation)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_seed)
    train_idx, val_idx = next(gss.split(X_df, y, groups=groups))
    
    X_train, y_train = X_df.iloc[train_idx], y[train_idx]
    X_val, y_val = X_df.iloc[val_idx], y[val_idx]
    val_pairs = [pair_ids[i] for i in val_idx]
    
    val_s1_ids = set(groups[i] for i in val_idx)
    val_ground_truth = {s1_id: ground_truth.get(s1_id, []) for s1_id in val_s1_ids}
    
    logger.info(
        f"Group Split: {len(train_idx):,} train pairs ({len(set(groups[i] for i in train_idx)):,} S1 entities) | "
        f"{len(val_idx):,} val pairs ({len(val_s1_ids):,} S1 entities)"
    )
    
    # Configure high-performance XGBoost
    scale_pos = (neg_count / max(pos_count, 1)) ** 0.5  # mild reweighting
    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=random_seed,
        n_jobs=-1
    )
    
    logger.info("Fitting XGBoost Classifier...")
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    
    # Validation Inference and F_0.5 Optimization
    val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold, val_f05 = optimize_decision_threshold(
        val_pairs, val_probs, val_ground_truth
    )
    
    # Log Top Feature Importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logger.info("Top Feature Importances:")
    for rank, idx in enumerate(sorted_idx[:8], 1):
        logger.info(f"  {rank}. {FEATURE_COLUMNS[idx]}: {importances[idx]:.4f}")
        
    # Save Model Artifacts
    model_path = os.path.join(model_output_dir, "xgboost_matcher.json")
    model.save_model(model_path)
    
    config_path = os.path.join(model_output_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold": best_threshold,
            "validation_f05": val_f05,
            "feature_columns": FEATURE_COLUMNS,
            "model_version": "1.0.0"
        }, f, indent=2)
        
    logger.info(f"Model and config successfully saved to {model_output_dir}/")
    return model, best_threshold
