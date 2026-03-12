"""
STEP 2: Interactive Semantic Code Search

A command-line interface for searching Python code with natural language queries.
Shows results from both semantic and keyword search for comparison.

Usage:
    python main_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from models.semantic_search import SemanticSearchEngine
from models.keyword_search import KeywordSearchEngine


def main():
    print("=" * 60)
    print("  SEMANTIC CODE SEARCH — Interactive Mode")
    print("=" * 60)
    
    # Load engines
    print("\nLoading search engines...")
    
    semantic = SemanticSearchEngine()
    semantic.load()
    
    keyword = KeywordSearchEngine()
    keyword.load()
    
    print("\nReady! Enter natural language queries to search Python code.")
    print("Commands: 'quit' to exit, 'compare' to toggle comparison mode")
    print("─" * 60)
    
    compare_mode = True
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if query.lower() == 'compare':
            compare_mode = not compare_mode
            print(f"Comparison mode: {'ON' if compare_mode else 'OFF'}")
            continue
        
        # ── Semantic Search ───────────────────────────────
        print(f"\n{'━' * 60}")
        print(f"  SEMANTIC SEARCH RESULTS")
        print(f"{'━' * 60}")
        
        results = semantic.search(query, top_k=5, verbose=False)
        
        for r in results:
            print(f"\n  [{r['rank']}] {r['func_name']}  "
                  f"(score: {r['score']:.4f}, lines: {r.get('code_lines', '?')})")
            
            if r.get('docstring'):
                doc = r['docstring'][:120].replace('\n', ' ')
                print(f"      Doc: {doc}")
            
            # Show first few lines of code
            code_lines = r['code'].split('\n')[:6]
            for line in code_lines:
                print(f"      {line}")
            if len(r['code'].split('\n')) > 6:
                print(f"      ...")
        
        # ── Keyword Search (comparison) ───────────────────
        if compare_mode:
            print(f"\n{'━' * 60}")
            print(f"  KEYWORD BASELINE RESULTS (TF-IDF)")
            print(f"{'━' * 60}")
            
            kw_results = keyword.search(query, top_k=5)
            
            for r in kw_results:
                print(f"\n  [{r['rank']}] {r['func_name']}  "
                      f"(score: {r['score']:.4f}, lines: {r.get('code_lines', '?')})")
                
                if r.get('docstring'):
                    doc = r['docstring'][:120].replace('\n', ' ')
                    print(f"      Doc: {doc}")


if __name__ == "__main__":
    main()
