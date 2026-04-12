"""
Keyword-based code search baseline using TF-IDF.

This serves as the baseline comparison for the semantic search approach.
Comparing semantic vs. keyword retrieval is a central finding of the project,
mirroring the evaluation strategy in Ryu et al. (2025).
"""

import pickle
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.preprocessing import preprocess_query, create_code_representation


class KeywordSearchEngine:
    """
    TF-IDF based keyword search baseline.
    
    Indexes function representations using TF-IDF vectors and retrieves
    results based on cosine similarity between query and document TF-IDF vectors.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            stop_words='english',
            sublinear_tf=True,  # Apply log normalization to TF
        )
        self.tfidf_matrix = None
        self.metadata: List[Dict] = []
    
    def build_index(self, functions: List[Dict]):
        """
        Build TF-IDF index from function metadata.
        
        Args:
            functions: List of function dicts with keys:
                func_name, docstring, code, code_lines
        """
        print(f"Building TF-IDF index for {len(functions)} functions...")
        
        self.metadata = functions
        
        # Create text representations for TF-IDF
        documents = []
        for func in functions:
            # Combine function name, docstring, and code tokens
            text = create_code_representation(
                func.get("func_name", ""),
                func.get("docstring", ""),
                func.get("code", ""),
            )
            # Include first 20 lines of code for keyword matching
            # Truncated to avoid inflating TF-IDF vocabulary with noise from large functions
            code_lines = func.get("code", "").split('\n')[:20]
            code_tokens = ' '.join(code_lines)
            combined = f"{text} {code_tokens}"
            documents.append(combined)
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search using TF-IDF cosine similarity.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of result dicts sorted by similarity score
        """
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        top_k = top_k or config.TOP_K_FINAL
        
        # Preprocess and vectorize query
        processed = preprocess_query(query)
        query_vec = self.vectorizer.transform([processed])
        
        # Compute cosine similarity against all documents
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            result = {
                **self.metadata[idx],
                "score": float(similarities[idx]),
                "rank": rank + 1,
                "index": int(idx),
            }
            results.append(result)
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = None) -> List[List[Dict]]:
        """Search for multiple queries."""
        top_k = top_k or config.TOP_K_FINAL
        
        processed = [preprocess_query(q) for q in queries]
        query_vecs = self.vectorizer.transform(processed)
        
        similarities = cosine_similarity(query_vecs, self.tfidf_matrix)
        
        all_results = []
        for q_idx, q_sims in enumerate(similarities):
            top_indices = np.argsort(q_sims)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices):
                result = {
                    **self.metadata[idx],
                    "score": float(q_sims[idx]),
                    "rank": rank + 1,
                    "index": int(idx),
                }
                results.append(result)
            
            all_results.append(results)
        
        return all_results
    
    def save(self, path: str = None):
        """Save the TF-IDF model and matrix."""
        path = path or str(config.INDEX_DIR / "tfidf_baseline.pkl")
        data = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "metadata": self.metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"TF-IDF baseline saved to {path}")
    
    def load(self, path: str = None):
        """Load a saved TF-IDF model."""
        path = path or str(config.INDEX_DIR / "tfidf_baseline.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.metadata = data["metadata"]
        print(f"TF-IDF baseline loaded. {self.tfidf_matrix.shape[0]} documents.")
