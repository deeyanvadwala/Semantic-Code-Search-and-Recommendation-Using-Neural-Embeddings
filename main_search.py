"""
STEP 2: Interactive Semantic Code Search

A command-line interface for searching Python code with natural language queries.

Internally uses Hybrid Search (semantic transformer + TF-IDF fused via
Reciprocal Rank Fusion) for better coverage than either engine alone.

Returns the single best matching function for each query.

Commands:
    <query>         — search with a natural language query
    index <path>    — add a .py file or directory of .py files to the index
    quit / q        — exit

Usage:
    python main_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from models.semantic_search import SemanticSearchEngine
from utils.code_parser import PythonFunctionExtractor

# Minimum RRF fusion score to display a result.
# Below this the query is considered unanswerable with the current index.
CONFIDENCE_THRESHOLD = 0.015

FALLBACK_MESSAGE = "We are currently working on it - will get back to you soon."


def _do_index(engine: HybridSearchEngine, path_str: str):
    """Parse a .py file or directory and add extracted functions to the index."""
    path = Path(path_str.strip())

    if not path.exists():
        print(f"  Path not found: {path}")
        return

    extractor = PythonFunctionExtractor(min_lines=2, require_docstring=False)

    if path.is_file():
        if path.suffix != ".py":
            print("  Only .py files are supported.")
            return
        functions = extractor.extract_from_file(str(path))
    else:
        functions = extractor.extract_from_directory(str(path), recursive=True)

    if not functions:
        print("  No functions found.")
        return

    print(f"  Found {len(functions)} functions — embedding and indexing...")
    added = engine.add_functions(functions)
    print(f"  Done. Added {added} functions to the search index.")


def main():
    print("=" * 60)
    print("  SEMANTIC CODE SEARCH — Interactive Mode")
    print("=" * 60)

    print("\nLoading search engine...")
    engine = SemanticSearchEngine()
    engine.load()

    print("\nReady! Enter a natural language query to search Python code.")
    print("  index <path>  — add your own .py files to the index")
    print("  quit          — exit")
    print("─" * 60)

    while True:
        try:
            raw = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue

        # ── Built-in commands ─────────────────────────────────
        if raw.lower() in ("quit", "exit", "q"):
            break

        if raw.lower().startswith("index "):
            _do_index(engine, raw[6:])
            continue

        # ── Search ────────────────────────────────────────────
        results = engine.search(raw, top_k=1)

        print()
        if not results or results[0]["score"] < CONFIDENCE_THRESHOLD:
            print(FALLBACK_MESSAGE)
        else:
            r = results[0]
            print(f"  {r['func_name']}")
            print()
            for line in r["code"].split("\n"):
                print(f"  {line}")


if __name__ == "__main__":
    main()
