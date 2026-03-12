"""
Transformer-based embedding model for encoding code and queries.

Uses multilingual-e5-large (Wang et al., 2024), validated by Ryu et al. (2025)
for semantic code search. The E5 model family requires specific prefixes:
  - "query: " for search queries
  - "passage: " for documents/code descriptions
"""

import numpy as np
import torch
from typing import List, Union
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class CodeEmbeddingModel:
    """Wrapper for transformer-based code/query embedding."""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = device or config.DEVICE
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading embedding model: {self.model_name}")
        print(f"Device: {self.device}")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.max_seq_length = config.MAX_SEQUENCE_LENGTH
        
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode_queries(self, queries: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode natural language queries into embeddings.
        
        E5 models require the "query: " prefix for search queries.
        
        Args:
            queries: List of natural language query strings
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_queries, embedding_dim)
        """
        # Add E5 query prefix
        prefixed = [f"{config.QUERY_PREFIX}{q}" for q in queries]
        
        embeddings = self.model.encode(
            prefixed,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def encode_passages(self, passages: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode code descriptions/passages into embeddings.
        
        E5 models require the "passage: " prefix for documents.
        
        Args:
            passages: List of code description strings
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_passages, embedding_dim)
        """
        # Add E5 passage prefix
        prefixed = [f"{config.PASSAGE_PREFIX}{p}" for p in passages]
        
        embeddings = self.model.encode(
            prefixed,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def encode_single_query(self, query: str) -> np.ndarray:
        """Encode a single query (convenience method)."""
        return self.encode_queries([query], show_progress=False)[0]
    
    def encode_single_passage(self, passage: str) -> np.ndarray:
        """Encode a single passage (convenience method)."""
        return self.encode_passages([passage], show_progress=False)[0]
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self.model.get_sentence_embedding_dimension()


# ─── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    model = CodeEmbeddingModel()
    
    # Test query encoding
    queries = [
        "sort a list of dictionaries by a specific key",
        "read a CSV file into a pandas dataframe",
    ]
    q_emb = model.encode_queries(queries)
    print(f"Query embeddings shape: {q_emb.shape}")
    
    # Test passage encoding
    passages = [
        "sort list dictionaries. Sorts a list of dictionaries by the specified key in ascending or descending order.",
        "read csv file. Reads a CSV file from the given path and returns a pandas DataFrame.",
    ]
    p_emb = model.encode_passages(passages)
    print(f"Passage embeddings shape: {p_emb.shape}")
    
    # Compute cosine similarity
    similarities = np.dot(q_emb, p_emb.T)
    print(f"\nCosine similarity matrix:")
    print(similarities)
    print("\n(Diagonal should be highest — matching query-passage pairs)")
