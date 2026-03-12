"""
Evaluation metrics for code search.

Implements the standard IR metrics used in Ryu et al. (2025):
- Recall@K (called SuccessRate@K in the paper)
- Precision@K
- MRR (Mean Reciprocal Rank)
- FRank (rank of first relevant result)
"""

import numpy as np
from typing import List, Dict, Optional


def recall_at_k(
    retrieved_ids: List[int], 
    relevant_ids: List[int], 
    k: int
) -> float:
    """
    Recall@K: fraction of relevant items found in top-K results.
    
    Equivalent to SuccessRate@K when there's exactly one relevant item
    (binary: 1 if found, 0 if not).
    
    Args:
        retrieved_ids: Ordered list of retrieved item IDs
        relevant_ids: Set of relevant item IDs
        k: Number of top results to consider
    """
    if not relevant_ids:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    
    return len(top_k & relevant) / len(relevant)


def success_rate_at_k(
    retrieved_ids: List[int], 
    relevant_ids: List[int], 
    k: int
) -> float:
    """
    SuccessRate@K: 1 if at least one relevant result is in top-K, else 0.
    
    This is the metric used in DeepCS and CodeMatcher papers.
    """
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    
    return 1.0 if len(top_k & relevant) > 0 else 0.0


def precision_at_k(
    retrieved_ids: List[int], 
    relevant_ids: List[int], 
    k: int
) -> float:
    """
    Precision@K: fraction of top-K results that are relevant.
    """
    if k == 0:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    
    return len(top_k & relevant) / k


def reciprocal_rank(
    retrieved_ids: List[int], 
    relevant_ids: List[int]
) -> float:
    """
    Reciprocal Rank: 1/rank of the first relevant result.
    
    Returns 0 if no relevant result is found.
    """
    relevant = set(relevant_ids)
    
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant:
            return 1.0 / (i + 1)
    
    return 0.0


def frank(
    retrieved_ids: List[int], 
    relevant_ids: List[int]
) -> Optional[int]:
    """
    FRank: rank of the first relevant result (1-indexed).
    
    Returns None if no relevant result found (NF in the paper).
    """
    relevant = set(relevant_ids)
    
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant:
            return i + 1
    
    return None


def evaluate_single_query(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k_values: List[int] = None,
) -> Dict:
    """
    Compute all metrics for a single query.
    
    Args:
        retrieved_ids: Ordered list of retrieved item IDs
        relevant_ids: List of relevant item IDs
        k_values: List of K values for @K metrics
        
    Returns:
        Dict with all metric values
    """
    k_values = k_values or [1, 3, 5, 10]
    
    results = {
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
        "frank": frank(retrieved_ids, relevant_ids),
    }
    
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"sr@{k}"] = success_rate_at_k(retrieved_ids, relevant_ids, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
    
    return results


def evaluate_all_queries(
    all_retrieved_ids: List[List[int]],
    all_relevant_ids: List[List[int]],
    k_values: List[int] = None,
) -> Dict:
    """
    Compute averaged metrics across all queries.
    
    Args:
        all_retrieved_ids: List of retrieved ID lists (one per query)
        all_relevant_ids: List of relevant ID lists (one per query)
        k_values: K values for @K metrics
        
    Returns:
        Dict with averaged metrics and per-query details
    """
    k_values = k_values or [1, 3, 5, 10]
    n_queries = len(all_retrieved_ids)
    
    per_query = []
    for retrieved, relevant in zip(all_retrieved_ids, all_relevant_ids):
        per_query.append(evaluate_single_query(retrieved, relevant, k_values))
    
    # Aggregate
    aggregated = {}
    
    # MRR
    aggregated["MRR"] = np.mean([pq["mrr"] for pq in per_query])
    
    # FRank statistics
    franks = [pq["frank"] for pq in per_query if pq["frank"] is not None]
    aggregated["FRank_mean"] = np.mean(franks) if franks else float('inf')
    aggregated["FRank_median"] = np.median(franks) if franks else float('inf')
    aggregated["FRank_found"] = len(franks)
    aggregated["FRank_not_found"] = n_queries - len(franks)
    
    # @K metrics
    for k in k_values:
        aggregated[f"Recall@{k}"] = np.mean([pq[f"recall@{k}"] for pq in per_query])
        aggregated[f"SR@{k}"] = np.mean([pq[f"sr@{k}"] for pq in per_query])
        aggregated[f"Precision@{k}"] = np.mean([pq[f"precision@{k}"] for pq in per_query])
    
    return {
        "aggregated": aggregated,
        "per_query": per_query,
        "n_queries": n_queries,
    }


def print_evaluation_report(eval_results: Dict, title: str = "Evaluation Results"):
    """Pretty-print evaluation results."""
    agg = eval_results["aggregated"]
    n = eval_results["n_queries"]
    
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  ({n} queries evaluated)")
    print(f"{'=' * 60}")
    
    print(f"\n  MRR:            {agg['MRR']:.4f}")
    print(f"  FRank (mean):   {agg['FRank_mean']:.2f}")
    print(f"  FRank (median): {agg['FRank_median']:.2f}")
    print(f"  Found/NF:       {agg['FRank_found']}/{agg['FRank_not_found']}")
    
    print(f"\n  {'K':>3}  {'SR@K':>8}  {'Recall@K':>10}  {'Precision@K':>12}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*10}  {'-'*12}")
    
    for key in sorted(agg.keys()):
        if key.startswith("SR@"):
            k = key.split("@")[1]
            print(f"  {k:>3}  {agg[f'SR@{k}']:>8.4f}  "
                  f"{agg[f'Recall@{k}']:>10.4f}  "
                  f"{agg[f'Precision@{k}']:>12.4f}")
    
    print(f"{'=' * 60}\n")
