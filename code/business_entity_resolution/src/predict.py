"""
Inference & Prediction Module
Applies trained XGBoost model and optimal decision threshold on candidate pairs.
"""

import os
import json
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .features import build_feature_matrix

logger = logging.getLogger(__name__)

def load_trained_matcher(model_dir: str = "models") -> tuple[XGBClassifier, float]:
    """
    Loads saved XGBoost model and configured optimal decision threshold.
    """
    model_path = os.path.join(model_dir, "xgboost_matcher.json")
    config_path = os.path.join(model_dir, "model_config.json")
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Model or config not found in {model_dir}")
        
    model = XGBClassifier()
    model.load_model(model_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    threshold = float(config.get("best_threshold", 0.60))
    logger.info(f"Loaded matcher from {model_path} (Decision Threshold: {threshold:.3f})")
    return model, threshold

def predict_matches(
    model: XGBClassifier,
    threshold: float,
    candidates_map: Dict[str, List[str]],
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """
    Runs inference on all candidate pairs and produces final match predictions.
    Guarantees:
    - Every S1 entity is present in returned dict
    - Predicted matches are a strict subset of candidate pairs
    - No duplicate target IDs per entity
    """
    predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_df["entity_id"]}
    
    logger.info(f"Running inference across candidate pairs for {len(s1_df):,} S1 entities...")
    X_features, pair_ids = build_feature_matrix(candidates_map, s1_df, targets_df)
    
    if len(pair_ids) == 0:
        logger.warning("No candidate pairs to score. All entities predicted as singletons.")
        return predictions
        
    probs = model.predict_proba(X_features)[:, 1]
    
    matched_count = 0
    for (s1_id, tgt_id), prob in zip(pair_ids, probs):
        if prob >= threshold:
            if tgt_id not in predictions[s1_id]:
                predictions[s1_id].append(tgt_id)
                matched_count += 1
                
    singletons = sum(1 for v in predictions.values() if len(v) == 0)
    logger.info(
        f"Inference complete: {matched_count:,} total matches predicted | "
        f"{singletons:,} singletons identified ({singletons/len(predictions):.1%})"
    )
    return predictions
