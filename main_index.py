"""
STEP 1: Build the Search Index

This script:
1. Loads the processed CodeSearchNet Python corpus
2. Creates text representations for each function
3. Encodes all functions using the transformer model
4. Builds a FAISS vector index
5. Builds a TF-IDF keyword baseline index
6. Saves both indexes to disk

Usage:
    python main_index.py
    
Expected time: ~15-30 min (depending on corpus size and GPU availability)
"""

import pickle
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils.preprocessing import create_code_representation
from models.embedding_model import CodeEmbeddingModel
from models.indexer import FAISSIndexer
from models.keyword_search import KeywordSearchEngine


def main():
    print("=" * 60)
    print("  SEMANTIC CODE SEARCH — INDEX BUILDER")
    print("=" * 60)
    
    # ── Load corpus ───────────────────────────────────────
    corpus_path = config.PROCESSED_DATA_DIR / "functions_corpus.pkl"
    
    if not corpus_path.exists():
        print(f"\nCorpus not found at {corpus_path}")
        print("Run 'python data/download_dataset.py' first!")
        sys.exit(1)
    
    print(f"\nLoading corpus from {corpus_path}...")
    with open(corpus_path, "rb") as f:
        functions = pickle.load(f)
    print(f"Loaded {len(functions)} functions")
    
    # ── Create text representations ───────────────────────
    print(f"\n{'─' * 40}")
    print("Creating text representations...")
    
    representations = []
    for func in functions:
        text = create_code_representation(
            func["func_name"],
            func["docstring"],
            func["code"],
        )
        representations.append(text)
        func["representation"] = text  # Store for later use
    
    print(f"Created {len(representations)} representations")
    print(f"Sample: '{representations[0][:100]}...'")
    
    # ── Encode with transformer ───────────────────────────
    print(f"\n{'─' * 40}")
    print("Encoding functions with transformer model...")
    
    model = CodeEmbeddingModel()
    
    t0 = time.time()
    embeddings = model.encode_passages(representations)
    encode_time = time.time() - t0
    
    print(f"Encoding completed in {encode_time:.1f}s")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Throughput: {len(functions)/encode_time:.0f} functions/sec")
    
    # ── Build FAISS index ─────────────────────────────────
    print(f"\n{'─' * 40}")
    print("Building FAISS index...")
    
    # Prepare metadata (stored alongside vectors)
    metadata = []
    for func in functions:
        metadata.append({
            "id": func["id"],
            "func_name": func["func_name"],
            "code": func["code"],
            "docstring": func["docstring"],
            "code_lines": func["code_lines"],
            "repo": func.get("repo", ""),
            "path": func.get("path", ""),
        })
    
    indexer = FAISSIndexer(embedding_dim=embeddings.shape[1])
    
    use_ivf = len(functions) > 10_000
    indexer.build_index(embeddings, metadata, use_ivf=use_ivf)
    indexer.save()
    
    # ── Build TF-IDF baseline ─────────────────────────────
    print(f"\n{'─' * 40}")
    print("Building TF-IDF keyword baseline...")
    
    keyword_engine = KeywordSearchEngine()
    keyword_engine.build_index(metadata)
    keyword_engine.save()
    
    # ── Summary ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  INDEX BUILDING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Functions indexed:  {len(functions)}")
    print(f"  Embedding dim:     {embeddings.shape[1]}")
    print(f"  FAISS index:       {config.FAISS_INDEX_PATH}")
    print(f"  TF-IDF index:      {config.INDEX_DIR / 'tfidf_baseline.pkl'}")
    print(f"  Encoding time:     {encode_time:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
