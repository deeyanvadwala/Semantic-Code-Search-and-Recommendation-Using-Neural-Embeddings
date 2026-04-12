"""
Hybrid Search Engine combining dense semantic retrieval and sparse TF-IDF
retrieval via Reciprocal Rank Fusion (RRF).

RRF formula (Cormack et al., 2009):
    score(d) = sum_i  1 / (k + rank_i(d))

where k = 60 is a smoothing constant that prevents very high-ranked documents
from dominating.  Results that appear in both result lists are boosted
naturally; results that only appear in one list still get partial credit.

Using both signals together compensates for the respective failure modes:
  - Semantic search can miss exact keyword matches.
  - Keyword (TF-IDF) search cannot handle synonyms or paraphrases.
"""

from typing import Dict, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.semantic_search import SemanticSearchEngine
from models.keyword_search import KeywordSearchEngine

# RRF smoothing constant — standard value from the original paper
RRF_K = 60


class HybridSearchEngine:
    """
    Fuses semantic (transformer + FAISS) and keyword (TF-IDF) rankings
    with Reciprocal Rank Fusion to return a single, merged result list.
    """

    def __init__(self):
        self.semantic = SemanticSearchEngine()
        self.keyword = KeywordSearchEngine()

    def load(self):
        """Load both underlying engines from disk."""
        self.semantic.load()
        self.keyword.load()
        return self

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Run both engines, fuse their rankings with RRF, and return top_k results.

        Args:
            query:  Natural language programming query
            top_k:  Number of final results to return

        Returns:
            List of result dicts sorted by RRF score (highest first).
            Each dict carries the same keys as SemanticSearchEngine.search(),
            with 'score' replaced by the RRF fusion score.
        """
        top_k = top_k or config.TOP_K_FINAL

        # Retrieve more candidates than needed so fusion has enough to work with
        n_candidates = config.TOP_K_CANDIDATES

        sem_results = self.semantic.search(query, top_k=n_candidates, apply_reranking=True)
        kw_results  = self.keyword.search(query,  top_k=n_candidates)

        # Accumulate RRF scores keyed by a stable function identity
        rrf_scores: Dict[str, float] = {}
        result_store: Dict[str, Dict] = {}

        for rank, r in enumerate(sem_results):
            key = _result_key(r)
            rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            result_store[key] = r

        for rank, r in enumerate(kw_results):
            key = _result_key(r)
            rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            if key not in result_store:
                result_store[key] = r

        # Sort by merged RRF score
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for i, key in enumerate(sorted_keys[:top_k]):
            entry = dict(result_store[key])
            entry["score"] = rrf_scores[key]
            entry["rank"]  = i + 1
            results.append(entry)

        return results

    def add_functions(self, functions: List[Dict]) -> int:
        """
        Proxy to SemanticSearchEngine.add_functions — adds new code to the
        FAISS index so it becomes searchable via both semantic and RRF paths.
        (The TF-IDF index is not updated incrementally; new functions are
        covered by the semantic half of the hybrid until the next full rebuild.)
        """
        return self.semantic.add_functions(functions)


def _result_key(r: Dict) -> str:
    """Stable deduplication key: function name + first line of its source."""
    first_line = r.get("code", "").split("\n")[0].strip()
    return f"{r.get('func_name', '')}::{first_line}"
