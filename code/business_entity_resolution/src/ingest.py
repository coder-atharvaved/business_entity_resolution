"""
Data Ingestion and Schema Validation Module
Safely loads multi-source TSV datasets with explicit tab-delimiter and strict schema checks.
"""

import os
import logging
from typing import Tuple, Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

def load_source_tsv(file_path: str, source_prefix: Optional[str] = None) -> pd.DataFrame:
    """
    Safely loads a source TSV file, validates the separator and columns,
    and handles missing fields without data corruption.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")
        
    logger.info(f"Loading source TSV: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    
    # Schema check
    for col in EXPECTED_SOURCE_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}' in {file_path}. Found columns: {list(df.columns)}. "
                "Ensure file is tab-separated."
            )
            
    # Verify entity prefix if specified
    if source_prefix and len(df) > 0:
        first_id = str(df["entity_id"].iloc[0])
        if not first_id.startswith(source_prefix):
            logger.warning(
                f"Expected prefix '{source_prefix}' for {file_path}, but found '{first_id}'"
            )
            
    # Clean whitespace
    for col in EXPECTED_SOURCE_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
        
    logger.info(f"Loaded {len(df):,} records from {os.path.basename(file_path)}")
    return df

def load_ground_truth(file_path: str) -> Dict[str, List[str]]:
    """
    Loads train ground truth TSV and converts it to a fast lookup mapping:
    {source1_id: [matched_s2_or_s3_id, ...]}
    Empty strings represent singleton entities (no matches).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
        
    logger.info(f"Loading ground truth: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    
    for col in EXPECTED_TRUTH_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in ground truth. Found: {list(df.columns)}")
            
    mapping: Dict[str, List[str]] = {}
    total_matches = 0
    singletons = 0
    
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = str(row["matched_entity_ids"]).strip()
        
        if not raw_matches:
            mapping[s1_id] = []
            singletons += 1
        else:
            matched_ids = [m.strip() for m in raw_matches.split(",") if m.strip()]
            mapping[s1_id] = matched_ids
            total_matches += len(matched_ids)
            
    logger.info(
        f"Ground truth parsed: {len(mapping):,} S1 entities | "
        f"{total_matches:,} true match pairs | {singletons:,} singletons ({singletons/len(mapping):.1%})"
    )
    return mapping

def load_dataset_split(
    data_dir: str, 
    split: str = "train"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, List[str]]]]:
    """
    Loads all source files for a given split ('train' or 'test').
    Returns: (s1_df, s2_df, s3_df, ground_truth_dict_or_None)
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        # Fallback if flat structure
        split_dir = data_dir
        
    s1_path = os.path.join(split_dir, f"{split}_source1.tsv")
    s2_path = os.path.join(split_dir, f"{split}_source2.tsv")
    s3_path = os.path.join(split_dir, f"{split}_source3.tsv")
    
    s1_df = load_source_tsv(s1_path, source_prefix="S1-")
    s2_df = load_source_tsv(s2_path, source_prefix="S2-")
    s3_df = load_source_tsv(s3_path, source_prefix="S3-")
    
    gt_mapping = None
    if split == "train":
        gt_path = os.path.join(split_dir, "train_ground_truth.tsv")
        if os.path.exists(gt_path):
            gt_mapping = load_ground_truth(gt_path)
            
    return s1_df, s2_df, s3_df, gt_mapping
