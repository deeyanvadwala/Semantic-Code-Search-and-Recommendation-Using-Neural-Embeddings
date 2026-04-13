"""
Evaluation runner: benchmarks Semantic search quality.

Metrics computed over PYTHON_QUERIES_50 (50 curated natural-language queries):

  MRR   — Mean Reciprocal Rank.  Average of 1/rank of the first relevant result.
           Higher = system finds a good answer earlier.

  R@K   — Recall at K.  Fraction of queries where at least one relevant result
           appears in the top K.  Reported for K in {1, 3, 5, 10}.

Relevance is approximated by keyword overlap between each query's keywords
and the function name + docstring of returned results.  This proxy is
reasonable for a benchmark without manual ground-truth labels.

Usage:
    python evaluation/run_eval.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.semantic_search import SemanticSearchEngine
from evaluation.queries import PYTHON_QUERIES_50


# ─── Metric helpers ──────────────────────────────────────────


def reciprocal_rank(results: list, relevant: set) -> float:
    """1 / rank of the first relevant result; 0 if none found."""
    for r in results:
        if r["func_name"] in relevant:
            return 1.0 / r["rank"]
    return 0.0


def recall_at_k(results: list, relevant: set, k: int) -> float:
    """1 if any of the top-k results is relevant, else 0."""
    top_k_names = {r["func_name"] for r in results[:k]}
    return 1.0 if top_k_names & relevant else 0.0


def _relevant_set(results: list, keywords: list) -> set:
    """
    Determine which returned functions are 'relevant' for a query.

    A result is considered relevant when at least half of the query's
    keywords appear in the function name or docstring.
    """
    threshold = max(1, len(keywords) // 2)
    relevant = set()
    for r in results:
        text = (r.get("func_name", "") + " " + r.get("docstring", "")).lower()
        if sum(1 for kw in keywords if kw in text) >= threshold:
            relevant.add(r["func_name"])
    return relevant


# ─── Main evaluation loop ─────────────────────────────────────


def run_evaluation():
    print("=" * 58)
    print("  EVALUATION: Semantic Search")
    print("=" * 58)

    print("\nLoading engine...")
    semantic = SemanticSearchEngine()
    semantic.load()

    K_VALUES = config.TOP_K_VALUES   # [1, 3, 5, 10]
    max_k    = max(K_VALUES)

    sem_mrr:    list = []
    sem_recall: dict = {k: [] for k in K_VALUES}

    skipped = 0

    print(f"\nRunning {len(PYTHON_QUERIES_50)} queries...\n")

    for entry in PYTHON_QUERIES_50:
        query    = entry["query"]
        keywords = entry["keywords"]

        sem_results = semantic.search(query, top_k=max_k)

        relevant = _relevant_set(sem_results, keywords)

        if not relevant:
            skipped += 1
            continue

        sem_mrr.append(reciprocal_rank(sem_results, relevant))

        for k in K_VALUES:
            sem_recall[k].append(recall_at_k(sem_results, relevant, k))

    n = len(sem_mrr)
    if n == 0:
        print("No evaluable queries found.")
        return

    # ─── Results table ────────────────────────────────────────
    col = 14
    print(f"{'Metric':<18} {'Semantic':>{col}}")
    print("─" * (18 + col + 2))

    def row(label, s_vals):
        s = sum(s_vals) / len(s_vals)
        print(f"{label:<18} {s:>{col}.4f}")

    row("MRR", sem_mrr)
    for k in K_VALUES:
        row(f"Recall@{k}", sem_recall[k])

    print(f"\nEvaluated on {n} / {len(PYTHON_QUERIES_50)} queries "
          f"({skipped} skipped — no relevant results found).")


if __name__ == "__main__":
    run_evaluation()
