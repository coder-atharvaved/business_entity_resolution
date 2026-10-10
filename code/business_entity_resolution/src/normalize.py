"""
Text Normalization & Preprocessing Module
Lightweight, memory-efficient string normalizer for multi-million record datasets.
"""

import re
import unicodedata
from typing import Set
import pandas as pd

LEGAL_SUFFIXES = [
    r"\bpvt\b", r"\bprivate\b", r"\bltd\b", r"\blimited\b", r"\bllc\b", r"\bl\.l\.c\b",
    r"\binc\b", r"\bincorporated\b", r"\bcorp\b", r"\bcorporation\b", r"\bco\b", r"\bcompany\b",
    r"\bsa\b", r"\bs\.a\b", r"\bsas\b", r"\bs\.a\.s\b", r"\bsarl\b", r"\bs\.a\.r\.l\b",
    r"\bgmbh\b", r"\benterprises?\b", r"\bservices?\b", r"\bsolutions?\b", r"\bassociates?\b",
    r"\bgroup\b", r"\bholdings?\b", r"\bpartners?\b", r"\btrust\b"
]
LEGAL_PATTERN = re.compile(r"|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

ADDRESS_REPLACEMENTS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
}
ADDRESS_PATTERNS = [(re.compile(p, flags=re.IGNORECASE), repl) for p, repl in ADDRESS_REPLACEMENTS.items()]

def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    text = strip_accents(name).lower()
    text = text.replace("&", " and ")
    text = LEGAL_PATTERN.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_address(address: str) -> str:
    if not isinstance(address, str) or not address.strip():
        return ""
    text = strip_accents(address).lower()
    for pattern, repl in ADDRESS_PATTERNS:
        text = pattern.sub(repl, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_numbers(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"\b\d+\b", text))

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds normalized fields as compact strings (without allocating millions of Python set objects).
    """
    df = df.copy()
    df["norm_name"] = df["business_name"].fillna("").astype(str).map(normalize_name)
    df["norm_addr"] = df["business_address"].fillna("").astype(str).map(normalize_address)
    df["country"] = df["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    return df
