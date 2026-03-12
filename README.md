# Semantic Code Search and Recommendation Using Neural Embeddings

**Author:** Deeyan Vadwala  
**Project Type:** Academic Research Project  
**Language Scope:** Python source code

## Project Overview

This system retrieves relevant Python code snippets in response to natural language 
programming queries by leveraging learned semantic representations (transformer-based 
neural embeddings) rather than relying on exact keyword matches.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INDEXING PIPELINE                  │
│                                                     │
│  Python Repos ──► AST Parser ──► Function Extractor │
│                                      │              │
│                              Function + Docstring   │
│                                      │              │
│                              Embedding Model        │
│                            (multilingual-e5-large)   │
│                                      │              │
│                              FAISS Vector Index     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   SEARCH PIPELINE                   │
│                                                     │
│  NL Query ──► Embedding Model ──► Cosine Similarity │
│                                        │            │
│                                  Re-ranking         │
│                              (length penalty +      │
│                               name similarity)      │
│                                        │            │
│                                  Top-K Results      │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
semantic_code_search/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── config.py                    # Central configuration
├── data/
│   ├── download_dataset.py      # Download CodeSearchNet dataset
│   └── raw/                     # Raw data storage
├── utils/
│   ├── __init__.py
│   ├── code_parser.py           # AST-based Python function extractor
│   └── preprocessing.py         # Text cleaning and normalization
├── models/
│   ├── __init__.py
│   ├── embedding_model.py       # Transformer embedding wrapper
│   ├── indexer.py               # FAISS index builder
│   ├── semantic_search.py       # Semantic search engine
│   └── keyword_search.py        # TF-IDF keyword baseline
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # Recall@K, MRR, Precision@K
│   ├── benchmark.py             # Run full evaluation pipeline
│   └── queries.py               # Curated evaluation queries
├── results/                     # Evaluation output
├── notebooks/
│   └── exploration.ipynb        # Data exploration notebook
├── main_index.py                # Step 1: Build index
├── main_search.py               # Step 2: Interactive search
└── main_evaluate.py             # Step 3: Run evaluation
```

## Setup & Execution

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
```bash
python data/download_dataset.py
```

### Step 3: Build the Index
```bash
python main_index.py
```

### Step 4: Run Interactive Search
```bash
python main_search.py
```

### Step 5: Run Evaluation
```bash
python main_evaluate.py
```

## Evaluation Metrics
- **Recall@K**: Fraction of queries where a relevant result appears in top-K
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant result
- **Precision@K**: Average fraction of relevant results in top-K

## Key Design Decisions (Informed by Ryu et al., 2025)
1. Shared embedding space for queries and code (multilingual-e5-large)
2. Docstrings as natural language proxy (instead of LLM-generated summaries)
3. FAISS for lightweight vector search (instead of Elasticsearch)
4. Length-based penalty in re-ranking (adapted from SEMANTIC CODE FINDER)
5. Function name similarity as secondary signal
