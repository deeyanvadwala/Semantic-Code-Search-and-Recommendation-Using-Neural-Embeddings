"""
FAISS-based vector index for efficient similarity search.

Replaces Elasticsearch (used in SEMANTIC CODE FINDER) with a 
lightweight in-memory solution suitable for research-scale projects.
"""

import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class FAISSIndexer:
    """Build and query a FAISS vector index for code embeddings."""
    
    def __init__(self, embedding_dim: int = None):
        """
        Args:
            embedding_dim: Dimensionality of embeddings
        """
        self.embedding_dim = embedding_dim or config.EMBEDDING_DIMENSION
        self.index = None
        self.metadata: List[Dict] = []  # Parallel metadata for each vector
    
    def build_index(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict],
        use_ivf: bool = False,
        n_clusters: int = 100,
    ):
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata: List of dicts with function metadata (parallel to embeddings)
            use_ivf: Whether to use IVF index (faster for large datasets)
            n_clusters: Number of clusters for IVF index
        """
        n_vectors, dim = embeddings.shape
        self.embedding_dim = dim
        self.metadata = metadata
        
        print(f"Building FAISS index: {n_vectors} vectors, {dim} dimensions")
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity (use inner product on normalized vectors)
        faiss.normalize_L2(embeddings)
        
        if use_ivf and n_vectors > 1000:
            # IVF index: faster search for large datasets
            n_clusters = min(n_clusters, n_vectors // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            
            # Train the index
            print(f"Training IVF index with {n_clusters} clusters...")
            self.index.train(embeddings)
            self.index.nprobe = config.NPROBE
        else:
            # Flat index: exact search (best for < 100K vectors)
            self.index = faiss.IndexFlatIP(dim)
        
        # Add vectors to index
        self.index.add(embeddings)

        # Move to GPU if requested
        if config.FAISS_USE_GPU and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("FAISS index moved to GPU.")

        print(f"Index built. Total vectors: {self.index.ntotal}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = None
    ) -> List[Tuple[int, float, Dict]]:
        """
        Search the index for the most similar vectors.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            top_k: Number of results to return
            
        Returns:
            List of (index, score, metadata) tuples, sorted by similarity
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        top_k = top_k or config.TOP_K_CANDIDATES
        
        # Reshape and normalize query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.index.search(query, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            results.append((int(idx), float(score), self.metadata[idx]))
        
        return results
    
    def batch_search(
        self, 
        query_embeddings: np.ndarray, 
        top_k: int = None
    ) -> List[List[Tuple[int, float, Dict]]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Array of shape (n_queries, embedding_dim)
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        top_k = top_k or config.TOP_K_CANDIDATES
        
        queries = query_embeddings.astype(np.float32)
        faiss.normalize_L2(queries)
        
        scores, indices = self.index.search(queries, top_k)
        
        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for score, idx in zip(q_scores, q_indices):
                if idx == -1:
                    continue
                results.append((int(idx), float(score), self.metadata[idx]))
            all_results.append(results)
        
        return all_results
    
    def save(self, index_path: str = None, metadata_path: str = None):
        """Save the index and metadata to disk."""
        index_path = index_path or str(config.FAISS_INDEX_PATH)
        metadata_path = metadata_path or str(config.METADATA_PATH)
        
        print(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        print(f"Saving metadata to {metadata_path}")
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        print("Index saved successfully.")
    
    def load(self, index_path: str = None, metadata_path: str = None):
        """Load the index and metadata from disk."""
        index_path = index_path or str(config.FAISS_INDEX_PATH)
        metadata_path = metadata_path or str(config.METADATA_PATH)
        
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"Index loaded. Total vectors: {self.index.ntotal}")
    
    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal if self.index else 0
