"""
STEP 3: Run Full Evaluation

Compares semantic search vs. keyword baseline on:
1. CodeSearchNet test pairs (docstring → function matching)
2. Curated Python Queries50 (qualitative analysis)

Produces:
- Metrics tables (MRR, SR@K, P@K, FRank)
- Comparison visualizations
- Per-query breakdown

Usage:
    python main_evaluate.py
"""

import pickle
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from models.semantic_search import SemanticSearchEngine
from models.keyword_search import KeywordSearchEngine
from evaluation.benchmark import BenchmarkRunner
from evaluation.queries import PYTHON_QUERIES_50


def main():
    print("=" * 60)
    print("  SEMANTIC CODE SEARCH — EVALUATION")
    print("=" * 60)
    
    # ── Load engines ──────────────────────────────────────
    print("\nLoading search engines...")
    
    semantic = SemanticSearchEngine()
    semantic.load()
    
    keyword = KeywordSearchEngine()
    keyword.load()
    
    # ── Part 1: Quantitative Evaluation ───────────────────
    print(f"\n{'=' * 60}")
    print("  PART 1: Quantitative Evaluation (CodeSearchNet Test Pairs)")
    print(f"{'=' * 60}")
    
    eval_path = config.PROCESSED_DATA_DIR / "eval_pairs.pkl"
    if eval_path.exists():
        with open(eval_path, "rb") as f:
            eval_pairs = pickle.load(f)
        
        print(f"Loaded {len(eval_pairs)} evaluation pairs")
        
        runner = BenchmarkRunner(semantic, keyword, eval_pairs)
        results = runner.run(
            n_queries=config.NUM_EVAL_QUERIES,
            k_values=config.TOP_K_VALUES,
            save_results=True,
        )
    else:
        print(f"Eval pairs not found at {eval_path}")
        print("Skipping quantitative evaluation.")
        results = None
    
    # ── Part 2: Qualitative Analysis ──────────────────────
    print(f"\n{'=' * 60}")
    print("  PART 2: Qualitative Analysis (Python Queries50)")
    print(f"{'=' * 60}")
    
    qualitative_results = []
    
    for q in PYTHON_QUERIES_50:
        print(f"\n{'─' * 50}")
        print(f"  Query {q['id']}: \"{q['query']}\"")
        print(f"  Category: {q['category']}")
        print(f"{'─' * 50}")
        
        # Semantic results
        sem_results = semantic.search(q["query"], top_k=3)
        print("\n  Semantic Top-3:")
        for r in sem_results:
            print(f"    [{r['rank']}] {r['func_name']} "
                  f"(score: {r['score']:.4f})")
        
        # Keyword results
        kw_results = keyword.search(q["query"], top_k=3)
        print("\n  Keyword Top-3:")
        for r in kw_results:
            print(f"    [{r['rank']}] {r['func_name']} "
                  f"(score: {r['score']:.4f})")
        
        qualitative_results.append({
            "query": q,
            "semantic_top3": [
                {"func_name": r["func_name"], "score": r["score"]} 
                for r in sem_results
            ],
            "keyword_top3": [
                {"func_name": r["func_name"], "score": r["score"]} 
                for r in kw_results
            ],
        })
    
    # Save qualitative results
    qual_path = config.RESULTS_DIR / "qualitative_analysis.json"
    with open(qual_path, "w") as f:
        json.dump(qualitative_results, f, indent=2)
    print(f"\nQualitative results saved to {qual_path}")
    
    # ── Part 3: Timing Analysis ───────────────────────────
    print(f"\n{'=' * 60}")
    print("  PART 3: Search Speed Comparison")
    print(f"{'=' * 60}")
    
    test_queries = [q["query"] for q in PYTHON_QUERIES_50[:20]]
    
    # Semantic timing
    t0 = time.time()
    for q in test_queries:
        semantic.search(q, top_k=10)
    sem_time = (time.time() - t0) / len(test_queries) * 1000
    
    # Keyword timing
    t0 = time.time()
    for q in test_queries:
        keyword.search(q, top_k=10)
    kw_time = (time.time() - t0) / len(test_queries) * 1000
    
    print(f"\n  Semantic search: {sem_time:.1f} ms/query")
    print(f"  Keyword search:  {kw_time:.1f} ms/query")
    
    # ── Summary ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Results directory: {config.RESULTS_DIR}")
    print(f"  Files generated:")
    for f in config.RESULTS_DIR.iterdir():
        print(f"    - {f.name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
