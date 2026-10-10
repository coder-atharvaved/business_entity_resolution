"""
Macro F_0.5 Metric & Decision Threshold Optimizer Module
Evaluates precision-heavy F_0.5 with exact singleton handling as specified by the evaluation criteria.
"""

import logging
from typing import Dict, List, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def compute_entity_f05(pred_matches: Set[str], true_matches: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity:
    - If true_matches is empty (singleton):
        scores 1.0 if pred_matches is empty, else 0.0.
    - If true_matches is non-empty:
        if pred_matches is empty, scores 0.0.
        otherwise computes Precision and Recall:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    n_true = len(true_matches)
    n_pred = len(pred_matches)
    
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
        
    if n_pred == 0:
        return 0.0
        
    correct = len(pred_matches.intersection(true_matches))
    if correct == 0:
        return 0.0
        
    precision = correct / n_pred
    recall = correct / n_true
    
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom

def compute_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]]
) -> float:
    """
    Computes macro-averaged F_0.5 across all Source 1 entities in ground_truth.
    """
    scores = []
    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        pred_set = set(predictions.get(s1_id, []))
        scores.append(compute_entity_f05(pred_set, true_set))
    return float(np.mean(scores)) if scores else 0.0

def optimize_decision_threshold(
    pair_ids: List[Tuple[str, str]],
    probabilities: np.ndarray,
    ground_truth: Dict[str, List[str]],
    min_thresh: float = 0.35,
    max_thresh: float = 0.90,
    steps: int = 30
) -> Tuple[float, float]:
    """
    Sweeps decision threshold cutoffs to identify the exact threshold maximizing macro F_0.5.
    Returns: (best_threshold, best_f05_score)
    """
    logger.info("Optimizing decision threshold for macro F_0.5...")
    best_thresh = 0.60
    best_score = -1.0
    
    threshold_grid = np.linspace(min_thresh, max_thresh, steps)
    
    for th in threshold_grid:
        preds: Dict[str, List[str]] = {s1_id: [] for s1_id in ground_truth}
        for (s1_id, tgt_id), prob in zip(pair_ids, probabilities):
            if prob >= th:
                if s1_id in preds:
                    preds[s1_id].append(tgt_id)
                    
        score = compute_macro_f05(preds, ground_truth)
        if score > best_score:
            best_score = score
            best_thresh = th
            
    logger.info(f"Optimal Threshold Found: {best_thresh:.3f} | Best Validation F_0.5: {best_score:.4f}")
    return float(best_thresh), float(best_score)
