"""
Download and prepare the CodeSearchNet Python dataset.

CodeSearchNet (Husain et al., 2019) contains Python functions paired with
their docstrings — a natural fit for semantic code search evaluation.

Each record contains:
  - func_code_string: the full function source code
  - func_documentation_string: the docstring
  - func_name: the function name
  - language: programming language (we filter for Python)
"""

import json
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def download_and_prepare():
    """Download CodeSearchNet Python data and extract function records."""
    
    from datasets import load_dataset
    
    print("=" * 60)
    print("STEP 1: Downloading CodeSearchNet Python Dataset")
    print("=" * 60)
    
    # Load the Python subset
    print("\nLoading dataset from HuggingFace Hub...")
    dataset = load_dataset(
        config.DATASET_NAME, 
        config.DATASET_LANGUAGE,
        trust_remote_code=True
    )
    
    print(f"\nDataset splits available: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} records")
    
    # ─── Process training split (largest) ─────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 2: Extracting Function Records")
    print(f"{'=' * 60}")
    
    functions = []
    skipped = 0
    
    # Use train split for the code corpus
    for i, record in enumerate(dataset["train"]):
        if len(functions) >= config.MAX_FUNCTIONS:
            break
        
        code = record.get("func_code_string", "").strip()
        docstring = record.get("func_documentation_string", "").strip()
        func_name = record.get("func_name", "").strip()
        
        # Skip functions without docstrings (we need NL descriptions)
        if not docstring or len(docstring) < 10:
            skipped += 1
            continue
        
        # Skip trivially short functions
        code_lines = len([l for l in code.split("\n") if l.strip()])
        if code_lines < 2:
            skipped += 1
            continue
        
        functions.append({
            "id": len(functions),
            "func_name": func_name,
            "code": code,
            "docstring": docstring,
            "code_lines": code_lines,
            "repo": record.get("repository_name", ""),
            "path": record.get("func_path_in_repo", ""),
        })
    
    print(f"\nExtracted: {len(functions)} functions")
    print(f"Skipped:   {skipped} (no docstring or too short)")
    
    # ─── Process test split for evaluation ────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 3: Preparing Evaluation Pairs")
    print(f"{'=' * 60}")
    
    eval_pairs = []
    for record in dataset["test"]:
        if len(eval_pairs) >= 500:  # Keep a pool for evaluation
            break
        
        code = record.get("func_code_string", "").strip()
        docstring = record.get("func_documentation_string", "").strip()
        func_name = record.get("func_name", "").strip()
        
        if not docstring or len(docstring) < 20:
            continue
        
        code_lines = len([l for l in code.split("\n") if l.strip()])
        if code_lines < 3:
            continue
            
        eval_pairs.append({
            "query": docstring,  # Use docstring as natural language query
            "relevant_code": code,
            "func_name": func_name,
            "code_lines": code_lines,
        })
    
    print(f"Evaluation pairs prepared: {len(eval_pairs)}")
    
    # ─── Save processed data ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 4: Saving Processed Data")
    print(f"{'=' * 60}")
    
    # Save functions corpus
    corpus_path = config.PROCESSED_DATA_DIR / "functions_corpus.pkl"
    with open(corpus_path, "wb") as f:
        pickle.dump(functions, f)
    print(f"Corpus saved: {corpus_path}")
    
    # Save evaluation pairs
    eval_path = config.PROCESSED_DATA_DIR / "eval_pairs.pkl"
    with open(eval_path, "wb") as f:
        pickle.dump(eval_pairs, f)
    print(f"Eval pairs saved: {eval_path}")
    
    # Save a sample for quick inspection
    sample_path = config.PROCESSED_DATA_DIR / "sample_functions.json"
    with open(sample_path, "w") as f:
        json.dump(functions[:5], f, indent=2)
    print(f"Sample saved: {sample_path}")
    
    # ─── Summary statistics ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    
    import numpy as np
    code_lengths = [f["code_lines"] for f in functions]
    doc_lengths = [len(f["docstring"].split()) for f in functions]
    
    print(f"Total functions in corpus:  {len(functions)}")
    print(f"Code lines - mean: {np.mean(code_lengths):.1f}, "
          f"median: {np.median(code_lengths):.1f}, "
          f"max: {np.max(code_lengths)}")
    print(f"Docstring words - mean: {np.mean(doc_lengths):.1f}, "
          f"median: {np.median(doc_lengths):.1f}, "
          f"max: {np.max(doc_lengths)}")
    print(f"\nTotal eval pairs: {len(eval_pairs)}")
    print(f"\nAll data saved to: {config.PROCESSED_DATA_DIR}")
    
    return functions, eval_pairs


if __name__ == "__main__":
    download_and_prepare()
