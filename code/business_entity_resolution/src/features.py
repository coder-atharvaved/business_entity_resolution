"""
Pairwise Feature Engineering Module
Computes 16 high-signal similarity features on demand for candidate pairs.
"""

import difflib
import re
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "name_tok_jaccard",
    "name_ng_jaccard",
    "name_seq_ratio",
    "name_len_diff",
    "name_exact_match",
    "name_containment",
    "name_first_tok_match",
    "addr_tok_jaccard",
    "addr_ng_jaccard",
    "addr_seq_ratio",
    "addr_len_diff",
    "num_jaccard",
    "num_common_count",
    "candidate_rank",
    "total_candidates",
    "target_is_s2",
]

def fast_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter_len = len(set_a.intersection(set_b))
    union_len = len(set_a) + len(set_b) - inter_len
    return inter_len / union_len if union_len > 0 else 0.0

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def extract_numbers(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"\b\d+\b", text))

def compute_pair_features(
    s1_row: Dict[str, Any],
    tgt_row: Dict[str, Any],
    candidate_rank: int,
    total_candidates: int
) -> Dict[str, float]:
    n1 = str(s1_row.get("norm_name", ""))
    n2 = str(tgt_row.get("norm_name", ""))
    a1 = str(s1_row.get("norm_addr", ""))
    a2 = str(tgt_row.get("norm_addr", ""))
    
    # 1. Name Features
    toks1 = set(n1.split()) if n1 else set()
    toks2 = set(n2.split()) if n2 else set()
    name_tok_jaccard = fast_jaccard(toks1, toks2)
    
    ng1 = get_char_ngrams(n1, 3)
    ng2 = get_char_ngrams(n2, 3)
    name_ng_jaccard = fast_jaccard(ng1, ng2)
    
    if n1 == n2:
        name_seq_ratio = 1.0
        name_exact_match = 1.0
        name_containment = 1.0
    else:
        name_seq_ratio = difflib.SequenceMatcher(None, n1, n2).quick_ratio() if (n1 and n2) else 0.0
        name_exact_match = 0.0
        name_containment = 1.0 if (n1 and n2 and (n1 in n2 or n2 in n1)) else 0.0
        
    name_len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
    first_tok_match = 1.0 if (n1 and n2 and n1.split()[:1] == n2.split()[:1]) else 0.0
    
    # 2. Address Features
    a_toks1 = set(a1.split()) if a1 else set()
    a_toks2 = set(a2.split()) if a2 else set()
    addr_tok_jaccard = fast_jaccard(a_toks1, a_toks2)
    
    a_ng1 = get_char_ngrams(a1, 3)
    a_ng2 = get_char_ngrams(a2, 3)
    addr_ng_jaccard = fast_jaccard(a_ng1, a_ng2)
    
    if a1 == a2:
        addr_seq_ratio = 1.0
    else:
        addr_seq_ratio = difflib.SequenceMatcher(None, a1, a2).quick_ratio() if (a1 and a2) else 0.0
        
    addr_len_diff = abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1)
    
    # 3. Numerical & Anchor Features
    num1 = extract_numbers(a1)
    num2 = extract_numbers(a2)
    num_jaccard = fast_jaccard(num1, num2)
    num_common_count = float(len(num1.intersection(num2)))
    
    # 4. Source Metadata
    target_is_s2 = 1.0 if str(tgt_row.get("entity_id", "")).startswith("S2-") else 0.0
    
    return {
        "name_tok_jaccard": name_tok_jaccard,
        "name_ng_jaccard": name_ng_jaccard,
        "name_seq_ratio": name_seq_ratio,
        "name_len_diff": name_len_diff,
        "name_exact_match": name_exact_match,
        "name_containment": name_containment,
        "name_first_tok_match": first_tok_match,
        "addr_tok_jaccard": addr_tok_jaccard,
        "addr_ng_jaccard": addr_ng_jaccard,
        "addr_seq_ratio": addr_seq_ratio,
        "addr_len_diff": addr_len_diff,
        "num_jaccard": num_jaccard,
        "num_common_count": num_common_count,
        "candidate_rank": float(candidate_rank),
        "total_candidates": float(total_candidates),
        "target_is_s2": target_is_s2,
    }

def build_feature_matrix(
    candidates_map: Dict[str, List[str]],
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    target_lookup = targets_df.set_index("entity_id").to_dict("index")
    
    pair_ids: List[Tuple[str, str]] = []
    feature_rows = []
    
    for s1_id, cand_list in candidates_map.items():
        s1_data = s1_lookup.get(s1_id, {})
        total_c = len(cand_list)
        for rank_idx, tgt_id in enumerate(cand_list, start=1):
            tgt_data = target_lookup.get(tgt_id, {})
            feats = compute_pair_features(s1_data, tgt_data, rank_idx, total_c)
            feature_rows.append(feats)
            pair_ids.append((s1_id, tgt_id))
            
    df_features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS) if feature_rows else pd.DataFrame(columns=FEATURE_COLUMNS)
    return df_features, pair_ids
