"""
Submission Writer Module
Serializes matching_results.tsv and candidate_pairs.tsv adhering strictly to official TSV formatting rules.
"""

import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def write_tsv_output(
    data_map: Dict[str, List[str]],
    s1_ordered_ids: List[str],
    output_path: str,
    id_column_name: str,
    list_column_name: str
) -> None:
    """
    Writes a TSV file with exact column headers and comma-separated IDs without quoting.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        # Header
        f.write(f"{id_column_name}\t{list_column_name}\n")
        
        for s1_id in s1_ordered_ids:
            target_ids = data_map.get(s1_id, [])
            # Deduplicate while preserving order
            seen = set()
            unique_targets = []
            for tid in target_ids:
                if tid not in seen and (tid.startswith("S2-") or tid.startswith("S3-")):
                    seen.add(tid)
                    unique_targets.append(tid)
                    
            joined_targets = ",".join(unique_targets)
            f.write(f"{s1_id}\t{joined_targets}\n")
            
    logger.info(f"Successfully wrote {len(s1_ordered_ids):,} rows to {output_path}")

def generate_submission_files(
    candidates_map: Dict[str, List[str]],
    predictions_map: Dict[str, List[str]],
    s1_ordered_ids: List[str],
    output_dir: str = "output"
) -> tuple[str, str]:
    """
    Generates both required submission files:
    1. output/matching_results.tsv
    2. output/candidate_pairs.tsv
    """
    os.makedirs(output_dir, exist_ok=True)
    
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    # 1. candidate_pairs.tsv
    write_tsv_output(
        data_map=candidates_map,
        s1_ordered_ids=s1_ordered_ids,
        output_path=candidate_path,
        id_column_name="source1_entity_id",
        list_column_name="candidate_entity_ids"
    )
    
    # 2. matching_results.tsv (ensuring matches are a subset of candidates)
    verified_predictions = {}
    for s1_id in s1_ordered_ids:
        cands_set = set(candidates_map.get(s1_id, []))
        preds = predictions_map.get(s1_id, [])
        # Strict subset enforcement
        subset_preds = [p for p in preds if p in cands_set]
        verified_predictions[s1_id] = subset_preds
        
    write_tsv_output(
        data_map=verified_predictions,
        s1_ordered_ids=s1_ordered_ids,
        output_path=matching_path,
        id_column_name="source1_entity_id",
        list_column_name="matched_entity_ids"
    )
    
    return matching_path, candidate_path
