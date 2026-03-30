"""
Central configuration for the Semantic Code Search project.
All hyperparameters, paths, and model settings in one place.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = PROJECT_ROOT / "indexes"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset Configuration ───────────────────────────────────
# Using CodeSearchNet Python subset (Husain et al., 2019)
DATASET_NAME = "code_search_net"
DATASET_LANGUAGE = "python"
MAX_FUNCTIONS = 50_000  # Limit for manageable compute; increase as needed
MAX_EVAL_PAIRS = 500

# Corpus quality filtering
MIN_DOCSTRING_CHARS = 10
MIN_EVAL_DOCSTRING_CHARS = 20
MIN_CODE_LINES = 2
MIN_EVAL_CODE_LINES = 3
MAX_CODE_LINES = 200
MAX_DOCSTRING_WORDS = 120
MIN_REPRESENTATION_TOKENS = 4

# Generic names create noisy matches for natural-language search.
GENERIC_FUNCTION_NAMES = {
    "add", "append", "apply", "build", "call", "check", "close", "copy",
    "create", "delete", "do", "dump", "execute", "fetch", "find", "from",
    "get", "handle", "init", "initialize", "insert", "load", "make", "open",
    "parse", "post", "process", "put", "read", "remove", "render", "run",
    "save", "send", "set", "sort", "start", "stop", "to", "update", "validate",
    "write",
}

# ─── Embedding Model ─────────────────────────────────────────
# multilingual-e5-large: validated by Ryu et al. (2025) for code search
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_DIMENSION = 1024  # Output dim for multilingual-e5-large
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32

# Query prefix required by E5 models
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# ─── FAISS Index ─────────────────────────────────────────────
FAISS_INDEX_PATH = INDEX_DIR / "code_embeddings.index"
METADATA_PATH = INDEX_DIR / "code_metadata.pkl"
FAISS_USE_GPU = False  # Set True if GPU available
NPROBE = 10  # Number of clusters to search (for IVF indexes)

# ─── Search Configuration ────────────────────────────────────
TOP_K_CANDIDATES = 50   # Initial retrieval pool (matching Ryu et al.)
TOP_K_FINAL = 10        # Final results after re-ranking
TOP_K_VALUES = [1, 3, 5, 10]  # K values for evaluation metrics

# ─── Re-ranking Configuration (adapted from SEMANTIC CODE FINDER) ─
# Penalty for trivially short functions.
# Keep penalties small — short utility functions (e.g. "add two numbers")
# are legitimate results and should not be strongly demoted.
SHORT_CODE_PENALTY_THRESHOLD_SEVERE = 2   # lines (only 1-2 line one-liners)
SHORT_CODE_PENALTY_SEVERE = 0.04          # 4% penalty (was 10%)
SHORT_CODE_PENALTY_THRESHOLD_MILD = 4     # lines
SHORT_CODE_PENALTY_MILD = 0.01            # 1% penalty (was 3%)

# Weight for function name similarity in re-ranking
NAME_SIMILARITY_WEIGHT = 0.10  # 10% contribution to final score

# ─── Keyword Baseline ────────────────────────────────────────
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# ─── Evaluation ──────────────────────────────────────────────
NUM_EVAL_QUERIES = 50
RANDOM_SEED = 42

# ─── Device Configuration ────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Config] Device: {DEVICE}")
print(f"[Config] Embedding model: {EMBEDDING_MODEL_NAME}")
print(f"[Config] Max functions: {MAX_FUNCTIONS}")
