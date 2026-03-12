"""
Full benchmark evaluation pipeline.

Compares semantic search vs. keyword baseline across all metrics,
generates result tables, and produces visualizations.
"""

import json
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from tabulate import tabulate

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from evaluation import (
    evaluate_single_query, evaluate_all_queries, print_evaluation_report
)


class BenchmarkRunner:
    """
    Run comparative evaluation of semantic vs. keyword search.
    
    Evaluation Strategy:
    - Use CodeSearchNet test set: docstrings as queries, matching functions as ground truth
    - For each query, check if the ground truth function (or semantically equivalent)
      appears in the top-K results
    - Compare semantic (transformer + FAISS) vs. keyword (TF-IDF) baselines
    """
    
    def __init__(self, semantic_engine, keyword_engine, eval_pairs: List[Dict]):
        """
        Args:
            semantic_engine: SemanticSearchEngine instance
            keyword_engine: KeywordSearchEngine instance
            eval_pairs: List of {query, relevant_code, func_name} dicts
        """
        self.semantic = semantic_engine
        self.keyword = keyword_engine
        self.eval_pairs = eval_pairs
    
    def run(
        self,
        n_queries: int = None,
        k_values: List[int] = None,
        save_results: bool = True,
    ) -> Dict:
        """
        Run the full benchmark.
        
        For each eval pair:
        1. Use the docstring as the query
        2. Search with both engines
        3. Check if the original function (matched by func_name + code similarity)
           appears in the results
        """
        n_queries = n_queries or min(config.NUM_EVAL_QUERIES, len(self.eval_pairs))
        k_values = k_values or config.TOP_K_VALUES
        
        eval_subset = self.eval_pairs[:n_queries]
        queries = [ep["query"] for ep in eval_subset]
        
        print(f"\n{'=' * 60}")
        print(f"  Running Benchmark: {n_queries} queries")
        print(f"  K values: {k_values}")
        print(f"{'=' * 60}\n")
        
        # ── Run Semantic Search ───────────────────────────────
        print("Running Semantic Search...")
        t0 = time.time()
        semantic_results = self.semantic.batch_search(
            queries, top_k=max(k_values)
        )
        semantic_time = time.time() - t0
        print(f"  Completed in {semantic_time:.2f}s "
              f"({semantic_time/n_queries*1000:.1f}ms per query)")
        
        # ── Run Keyword Search ────────────────────────────────
        print("Running Keyword Search...")
        t0 = time.time()
        keyword_results = self.keyword.batch_search(
            queries, top_k=max(k_values)
        )
        keyword_time = time.time() - t0
        print(f"  Completed in {keyword_time:.2f}s "
              f"({keyword_time/n_queries*1000:.1f}ms per query)")
        
        # ── Evaluate ──────────────────────────────────────────
        print("\nEvaluating results...")
        
        semantic_metrics = self._evaluate_results(
            eval_subset, semantic_results, k_values
        )
        keyword_metrics = self._evaluate_results(
            eval_subset, keyword_results, k_values
        )
        
        # ── Print Results ─────────────────────────────────────
        print_evaluation_report(semantic_metrics, "Semantic Search (Transformer + FAISS)")
        print_evaluation_report(keyword_metrics, "Keyword Baseline (TF-IDF)")
        
        # ── Comparison Table ──────────────────────────────────
        self._print_comparison(semantic_metrics, keyword_metrics, k_values)
        
        # ── Save Results ──────────────────────────────────────
        results = {
            "semantic": semantic_metrics,
            "keyword": keyword_metrics,
            "timing": {
                "semantic_total": semantic_time,
                "keyword_total": keyword_time,
                "semantic_per_query_ms": semantic_time / n_queries * 1000,
                "keyword_per_query_ms": keyword_time / n_queries * 1000,
            },
            "config": {
                "n_queries": n_queries,
                "k_values": k_values,
                "model": config.EMBEDDING_MODEL_NAME,
            }
        }
        
        if save_results:
            self._save_results(results)
            self._generate_plots(semantic_metrics, keyword_metrics, k_values)
        
        return results
    
    def _evaluate_results(
        self, 
        eval_pairs: List[Dict], 
        search_results: List[List[Dict]],
        k_values: List[int],
    ) -> Dict:
        """
        Evaluate search results against ground truth.
        
        Matching strategy: a result is "relevant" if its function name 
        matches the ground truth function name (flexible matching).
        """
        all_retrieved = []
        all_relevant = []
        
        for ep, results in zip(eval_pairs, search_results):
            gt_name = ep["func_name"].lower().strip()
            gt_code = ep["relevant_code"].strip()
            
            retrieved_ids = []
            relevant_ids = []
            
            for i, res in enumerate(results):
                retrieved_ids.append(i)
                
                # Match by function name
                res_name = res.get("func_name", "").lower().strip()
                
                # Flexible matching: exact name match or code similarity
                is_relevant = (
                    res_name == gt_name or
                    _code_similarity(res.get("code", ""), gt_code) > 0.8
                )
                
                if is_relevant:
                    relevant_ids.append(i)
            
            all_retrieved.append(retrieved_ids)
            # If no match found in results, relevant IDs stay empty
            # which correctly gives SR@K = 0
            all_relevant.append(relevant_ids if relevant_ids else [-1])
        
        return evaluate_all_queries(all_retrieved, all_relevant, k_values)
    
    def _print_comparison(self, semantic: Dict, keyword: Dict, k_values: List[int]):
        """Print side-by-side comparison table."""
        sem = semantic["aggregated"]
        kw = keyword["aggregated"]
        
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON: Semantic vs. Keyword Baseline")
        print(f"{'=' * 70}")
        
        rows = [
            ["MRR", f"{sem['MRR']:.4f}", f"{kw['MRR']:.4f}", 
             f"{(sem['MRR']-kw['MRR'])/max(kw['MRR'],0.001)*100:+.1f}%"],
            ["FRank (mean)", f"{sem['FRank_mean']:.2f}", f"{kw['FRank_mean']:.2f}", 
             f"{(kw['FRank_mean']-sem['FRank_mean'])/max(kw['FRank_mean'],0.001)*100:+.1f}%"],
        ]
        
        for k in k_values:
            rows.append([
                f"SR@{k}",
                f"{sem[f'SR@{k}']:.4f}",
                f"{kw[f'SR@{k}']:.4f}",
                f"{(sem[f'SR@{k}']-kw[f'SR@{k}'])/max(kw[f'SR@{k}'],0.001)*100:+.1f}%",
            ])
        
        for k in k_values:
            rows.append([
                f"P@{k}",
                f"{sem[f'Precision@{k}']:.4f}",
                f"{kw[f'Precision@{k}']:.4f}",
                f"{(sem[f'Precision@{k}']-kw[f'Precision@{k}'])/max(kw[f'Precision@{k}'],0.001)*100:+.1f}%",
            ])
        
        headers = ["Metric", "Semantic", "Keyword", "Improvement"]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    def _save_results(self, results: Dict):
        """Save evaluation results to disk."""
        # Save as JSON (convert numpy types)
        results_json = _convert_to_serializable(results)
        
        json_path = config.RESULTS_DIR / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {json_path}")
    
    def _generate_plots(self, semantic: Dict, keyword: Dict, k_values: List[int]):
        """Generate comparison visualizations."""
        sem = semantic["aggregated"]
        kw = keyword["aggregated"]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: SR@K comparison
        ax = axes[0]
        sr_sem = [sem[f"SR@{k}"] for k in k_values]
        sr_kw = [kw[f"SR@{k}"] for k in k_values]
        x = np.arange(len(k_values))
        width = 0.35
        ax.bar(x - width/2, sr_sem, width, label='Semantic', color='#2196F3')
        ax.bar(x + width/2, sr_kw, width, label='Keyword', color='#FF9800')
        ax.set_xlabel('K')
        ax.set_ylabel('Success Rate')
        ax.set_title('SuccessRate@K Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Plot 2: Precision@K comparison
        ax = axes[1]
        p_sem = [sem[f"Precision@{k}"] for k in k_values]
        p_kw = [kw[f"Precision@{k}"] for k in k_values]
        ax.bar(x - width/2, p_sem, width, label='Semantic', color='#2196F3')
        ax.bar(x + width/2, p_kw, width, label='Keyword', color='#FF9800')
        ax.set_xlabel('K')
        ax.set_ylabel('Precision')
        ax.set_title('Precision@K Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Plot 3: MRR comparison
        ax = axes[2]
        methods = ['Semantic\n(Ours)', 'Keyword\n(TF-IDF)']
        mrr_values = [sem['MRR'], kw['MRR']]
        colors = ['#2196F3', '#FF9800']
        bars = ax.bar(methods, mrr_values, color=colors, width=0.5)
        ax.set_ylabel('MRR')
        ax.set_title('Mean Reciprocal Rank')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, mrr_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = config.RESULTS_DIR / "benchmark_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plots saved to {plot_path}")


def _code_similarity(code1: str, code2: str) -> float:
    """Simple token-level similarity between two code snippets."""
    tokens1 = set(code1.split())
    tokens2 = set(code2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def _convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
