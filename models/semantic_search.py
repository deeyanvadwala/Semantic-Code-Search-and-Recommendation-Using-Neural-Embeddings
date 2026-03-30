"""
Semantic Search Engine — Core retrieval + re-ranking pipeline.

Implements the full search pipeline:
1. Encode query using transformer model
2. Retrieve top-K candidates via FAISS
3. Re-rank using length penalty + function name similarity
   (adapted from Ryu et al., 2025 — SEMANTIC CODE FINDER)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.embedding_model import CodeEmbeddingModel
from models.indexer import FAISSIndexer
from utils.preprocessing import preprocess_query, normalize_function_name


class SemanticSearchEngine:
    """
    Full semantic code search pipeline with re-ranking.
    
    Architecture:
        Query -> Embed -> FAISS Retrieval (top-50) -> Re-rank -> Top-10
    
    Re-ranking factors (from Ryu et al., 2025):
        1. Length penalty: penalize trivially short functions
        2. Name similarity: boost functions whose names match the query
    """
    
    def __init__(
        self,
        embedding_model: Optional[CodeEmbeddingModel] = None,
        indexer: Optional[FAISSIndexer] = None,
    ):
        self.embedding_model = embedding_model
        self.indexer = indexer
    
    def load(self):
        """Load pre-built model and index."""
        if self.embedding_model is None:
            self.embedding_model = CodeEmbeddingModel()
        if self.indexer is None:
            self.indexer = FAISSIndexer()
            self.indexer.load()
        return self
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        apply_reranking: bool = True,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Search for relevant Python functions given a natural language query.
        
        Args:
            query: Natural language programming query
            top_k: Number of final results to return
            apply_reranking: Whether to apply re-ranking heuristics
            verbose: Print debug information
            
        Returns:
            List of result dicts with keys:
                - func_name, code, docstring, code_lines
                - score (final re-ranked score)
                - rank (1-indexed position)
        """
        top_k = top_k or config.TOP_K_FINAL
        
        # Step 1: Preprocess query
        processed_query = preprocess_query(query)
        if verbose:
            print(f"Processed query: '{processed_query}'")
        
        # Step 2: Encode query
        query_embedding = self.embedding_model.encode_single_query(processed_query)
        
        # Step 3: Retrieve candidates from FAISS
        n_candidates = config.TOP_K_CANDIDATES if apply_reranking else top_k
        candidates = self.indexer.search(query_embedding, top_k=n_candidates)
        
        if verbose:
            print(f"Retrieved {len(candidates)} candidates from index")
        
        # Step 4: Re-rank (if enabled)
        if apply_reranking:
            results = self._rerank(processed_query, candidates, top_k, verbose)
        else:
            results = [
                {**meta, "score": score, "index": idx}
                for idx, score, meta in candidates[:top_k]
            ]
        
        # Add rank
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        return results
    
    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float, Dict]],
        top_k: int,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Re-rank candidates using length penalty and name similarity.
        
        Adapted from SEMANTIC CODE FINDER (Ryu et al., 2025):
        
        Step 1: Apply length penalty to cosine similarity scores
            - code_lines <= 3: 10% penalty
            - code_lines <= 5: 3% penalty
            
        Step 2: Take top 2*K candidates after penalty
        
        Step 3: Add function name similarity bonus (10% weight)
        
        Step 4: Return final top-K
        """
        # ── Step 1: Length penalty ────────────────────────────
        penalized = []
        for idx, score, meta in candidates:
            code_lines = meta.get("code_lines", 10)
            
            if code_lines <= config.SHORT_CODE_PENALTY_THRESHOLD_SEVERE:
                adjusted = score * (1 - config.SHORT_CODE_PENALTY_SEVERE)
            elif code_lines <= config.SHORT_CODE_PENALTY_THRESHOLD_MILD:
                adjusted = score * (1 - config.SHORT_CODE_PENALTY_MILD)
            else:
                adjusted = score
            
            penalized.append((idx, adjusted, meta))
        
        # ── Step 2: Keep top 2*K after penalty ────────────────
        penalized.sort(key=lambda x: x[1], reverse=True)
        shortlist = penalized[:top_k * 2]
        
        if verbose:
            print(f"After length penalty: shortlisted {len(shortlist)} candidates")
        
        # ── Step 3: Name similarity bonus + primary-token signal ──
        query_tokens = set(query.lower().split())
        query_words = query.lower().split()
        # First word is usually the operation/verb ("add", "sort", "read", …)
        primary_token = query_words[0] if query_words else ""

        final_scored = []
        for idx, score, meta in shortlist:
            func_name = meta.get("func_name", "")
            name_readable = normalize_function_name(func_name)
            name_tokens = set(name_readable.split())

            # Token overlap between query and function name
            if name_tokens and query_tokens:
                overlap = len(query_tokens & name_tokens)
                name_sim = overlap / max(len(query_tokens), len(name_tokens))
            else:
                name_sim = 0.0

            # Penalize results where the primary query token (the operation) is
            # absent from both the function name and docstring.
            # Fuzzy match handles "adds"/"adding" satisfying query token "add".
            docstring = meta.get("docstring", "").lower()
            doc_tokens = set(docstring.split())
            all_tokens = name_tokens | doc_tokens
            has_primary_signal = _fuzzy_token_match(primary_token, all_tokens)
            keyword_penalty = 0.0 if has_primary_signal else 0.05

            # Final score: cosine + name bonus - noise penalty
            final_score = score + (config.NAME_SIMILARITY_WEIGHT * name_sim) - keyword_penalty

            result = {**meta, "score": final_score, "index": idx}
            if verbose:
                result["_debug"] = {
                    "cosine_score": score,
                    "name_sim": name_sim,
                    "name_bonus": config.NAME_SIMILARITY_WEIGHT * name_sim,
                    "primary_token": primary_token,
                    "has_primary_signal": has_primary_signal,
                    "keyword_penalty": keyword_penalty,
                }
            final_scored.append(result)

        # ── Step 4: Final ranking ─────────────────────────────
        final_scored.sort(key=lambda x: x["score"], reverse=True)

        return final_scored[:top_k]
    
    def batch_search(
        self, 
        queries: List[str], 
        top_k: int = None,
        apply_reranking: bool = True,
    ) -> List[List[Dict]]:
        """Search for multiple queries (used in evaluation)."""
        top_k = top_k or config.TOP_K_FINAL
        
        # Encode all queries at once (more efficient)
        processed = [preprocess_query(q) for q in queries]
        query_embeddings = self.embedding_model.encode_queries(processed)
        
        # Retrieve candidates in batch
        n_candidates = config.TOP_K_CANDIDATES if apply_reranking else top_k
        all_candidates = self.indexer.batch_search(query_embeddings, top_k=n_candidates)
        
        # Re-rank each result set
        all_results = []
        for query, candidates in zip(processed, all_candidates):
            if apply_reranking:
                results = self._rerank(query, candidates, top_k)
            else:
                results = [
                    {**meta, "score": score, "index": idx}
                    for idx, score, meta in candidates[:top_k]
                ]
            
            for i, r in enumerate(results):
                r["rank"] = i + 1
            
            all_results.append(results)
        
        return all_results
    
    def format_result(self, result: Dict, show_code: bool = True) -> str:
        """Pretty-print a single search result."""
        lines = [
            f"[Rank {result['rank']}] {result['func_name']}  (score: {result['score']:.4f})",
            f"  Lines: {result.get('code_lines', '?')} | "
            f"Repo: {result.get('repo', 'N/A')}",
        ]
        
        if result.get('docstring'):
            doc_preview = result['docstring'][:100]
            lines.append(f"  Docstring: {doc_preview}...")
        
        if show_code:
            code_preview = '\n'.join(result['code'].split('\n')[:8])
            lines.append(f"  Code:\n{_indent(code_preview, 4)}")
        
        if '_debug' in result:
            d = result['_debug']
            lines.append(
                f"  [Debug] cosine={d['cosine_score']:.4f}, "
                f"name_sim={d['name_sim']:.3f}, "
                f"name_bonus={d['name_bonus']:.4f}"
            )
        
        return '\n'.join(lines)


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text."""
    prefix = ' ' * spaces
    return '\n'.join(prefix + line for line in text.split('\n'))


def _fuzzy_token_match(token: str, token_set: set) -> bool:
    """Check if token matches any element in token_set with light morphological fuzzing.

    Exact match is always tried first. For tokens >= 3 chars, also checks if any
    set member starts with the token and extends by at most 3 characters — this
    handles common inflections: add→adds/added/adding, sort→sorted/sorting, etc.
    """
    if token in token_set:
        return True
    if len(token) >= 3:
        return any(
            t.startswith(token) and 0 < len(t) - len(token) <= 3
            for t in token_set
        )
    return False
