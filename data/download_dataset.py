"""
Download and prepare the code search corpus from three datasets:

1. CodeSearchNet (Husain et al., 2019) — 50K real-world Python functions from GitHub
2. MBPP (Austin et al., 2021)          — ~400 basic Python programming problems
3. HumanEval (Chen et al., 2021)       — 164 clean, hand-crafted Python functions

MBPP and HumanEval fill the gap left by CodeSearchNet: simple utility functions
like "add two numbers" that real GitHub projects rarely document properly.
"""

import ast
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.preprocessing import evaluate_corpus_quality


# ─── CodeSearchNet ────────────────────────────────────────────

def _load_codesearchnet(dataset) -> Tuple[List[Dict], List[Dict], Dict]:
    """Extract corpus functions and eval pairs from CodeSearchNet."""
    functions = []
    eval_pairs = []
    skipped = {}

    # Training split → corpus
    for record in dataset["train"]:
        if len(functions) >= config.MAX_FUNCTIONS:
            break

        code      = record.get("func_code_string", "").strip()
        docstring = record.get("func_documentation_string", "").strip()
        func_name = record.get("func_name", "").strip()
        code_lines = len([l for l in code.split("\n") if l.strip()])

        keep, reason, quality = evaluate_corpus_quality(
            func_name, docstring, code, code_lines, eval_mode=False
        )
        if not keep:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        functions.append({
            "id": len(functions),
            "func_name": func_name,
            "normalized_name": quality["normalized_name"],
            "code": code,
            "docstring": docstring,
            "code_lines": code_lines,
            "representation": quality["representation"],
            "repo": record.get("repository_name", ""),
            "path": record.get("func_path_in_repo", ""),
        })

    # Test split → eval pairs
    eval_skipped = {}
    for record in dataset["test"]:
        if len(eval_pairs) >= config.MAX_EVAL_PAIRS:
            break

        code      = record.get("func_code_string", "").strip()
        docstring = record.get("func_documentation_string", "").strip()
        func_name = record.get("func_name", "").strip()
        code_lines = len([l for l in code.split("\n") if l.strip()])

        keep, reason, quality = evaluate_corpus_quality(
            func_name, docstring, code, code_lines, eval_mode=True
        )
        if not keep:
            eval_skipped[reason] = eval_skipped.get(reason, 0) + 1
            continue

        eval_pairs.append({
            "query": docstring,
            "relevant_code": code,
            "func_name": func_name,
            "normalized_name": quality["normalized_name"],
            "representation": quality["representation"],
            "code_lines": code_lines,
        })

    return functions, eval_pairs, skipped


# ─── MBPP helpers ─────────────────────────────────────────────

def _mbpp_clean_text(text: str) -> str:
    """Turn 'Write a function to X' into a clean docstring 'X'."""
    text = text.strip()
    # Remove common MBPP preambles
    text = re.sub(
        r"^Write\s+a?\s*(?:python\s+)?function\s+(?:to\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Capitalise the first letter
    if text:
        text = text[0].upper() + text[1:]
    # Strip trailing period/whitespace
    text = text.rstrip(". ") + "."
    return text


def _extract_main_function(code: str) -> Optional[Dict]:
    """Return the last top-level function in the code (MBPP puts the solution last)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.col_offset == 0          # top-level only
    ]
    if not funcs:
        return None

    node = funcs[-1]                   # last top-level function is the solution
    func_lines = code.splitlines()
    start = node.lineno - 1
    # Collect until the next top-level def or end of file
    end = len(func_lines)
    for other in funcs:
        if other.lineno > node.lineno and other.col_offset == 0:
            end = other.lineno - 1
            break

    func_code = "\n".join(func_lines[start:end]).strip()
    code_lines = len([l for l in func_code.splitlines() if l.strip()])
    return {"func_name": node.name, "code": func_code, "code_lines": code_lines}


def _load_mbpp(id_offset: int) -> Tuple[List[Dict], Dict]:
    """Load MBPP and return corpus-ready function records."""
    from datasets import load_dataset

    print("\nLoading MBPP dataset...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")

    # Detect field names from the first available record
    first_split = next(iter(ds.values()))
    if len(first_split) > 0:
        sample = first_split[0]
        available = list(sample.keys())
        print(f"  MBPP fields: {available}")
        # Support both old and new field naming conventions
        code_field = "code" if "code" in available else "source_code"
        # MBPP uses "prompt" for the problem description; older versions used "text"
        if "text" in available:
            text_field = "text"
        elif "prompt" in available:
            text_field = "prompt"
        else:
            text_field = "description"
    else:
        code_field, text_field = "code", "text"

    functions = []
    skipped = {}

    for split_name in ("train", "validation", "test", "prompt"):
        if split_name not in ds:
            continue
        for record in ds[split_name]:
            raw_code = record.get(code_field, "").strip()
            text     = record.get(text_field, "").strip()

            if not raw_code or not text:
                skipped["missing_fields"] = skipped.get("missing_fields", 0) + 1
                continue

            extracted = _extract_main_function(raw_code)
            if not extracted:
                skipped["no_function_found"] = skipped.get("no_function_found", 0) + 1
                continue

            docstring = _mbpp_clean_text(text)
            func_name = extracted["func_name"]
            code      = extracted["code"]
            code_lines = extracted["code_lines"]

            keep, reason, quality = evaluate_corpus_quality(
                func_name, docstring, code, code_lines, eval_mode=False
            )
            if not keep:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue

            functions.append({
                "id": id_offset + len(functions),
                "func_name": func_name,
                "normalized_name": quality["normalized_name"],
                "code": code,
                "docstring": docstring,
                "code_lines": code_lines,
                "representation": quality["representation"],
                "repo": "mbpp/google-research-datasets",
                "path": f"mbpp/task_{record.get('task_id', len(functions))}.py",
            })

    return functions, skipped


# ─── HumanEval helpers ────────────────────────────────────────

def _reconstruct_humaneval_function(prompt: str, solution: str) -> Optional[str]:
    """Combine prompt (def + docstring) and canonical solution into one function."""
    # prompt ends just before the body; solution is indented body lines
    full = prompt.rstrip() + "\n" + solution
    # Verify it parses
    try:
        ast.parse(full)
        return full.strip()
    except SyntaxError:
        return None


def _extract_docstring_from_code(code: str) -> str:
    """Pull the docstring out of a parsed function."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return ast.get_docstring(node) or ""
    except SyntaxError:
        pass
    return ""


def _load_humaneval(id_offset: int) -> Tuple[List[Dict], Dict]:
    """Load HumanEval and return corpus-ready function records."""
    from datasets import load_dataset

    print("\nLoading HumanEval dataset...")
    ds = load_dataset("openai_humaneval")

    functions = []
    skipped = {}

    for record in ds["test"]:
        prompt    = record.get("prompt", "")
        solution  = record.get("canonical_solution", "")
        func_name = record.get("entry_point", "").strip()

        if not prompt or not solution or not func_name:
            skipped["missing_fields"] = skipped.get("missing_fields", 0) + 1
            continue

        code = _reconstruct_humaneval_function(prompt, solution)
        if not code:
            skipped["syntax_error"] = skipped.get("syntax_error", 0) + 1
            continue

        docstring  = _extract_docstring_from_code(code)
        code_lines = len([l for l in code.splitlines() if l.strip()])

        keep, reason, quality = evaluate_corpus_quality(
            func_name, docstring, code, code_lines, eval_mode=False
        )
        if not keep:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        functions.append({
            "id": id_offset + len(functions),
            "func_name": func_name,
            "normalized_name": quality["normalized_name"],
            "code": code,
            "docstring": docstring,
            "code_lines": code_lines,
            "representation": quality["representation"],
            "repo": "humaneval/openai",
            "path": f"humaneval/{record.get('task_id', func_name)}.py",
        })

    return functions, skipped


# ─── LeetCode helpers ─────────────────────────────────────────

def _load_leetcode(id_offset: int) -> Tuple[List[Dict], Dict]:
    """Load LeetCode Python solutions and return corpus-ready records."""
    from datasets import load_dataset

    print("\nLoading LeetCode dataset...")
    try:
        ds = load_dataset("mhhmm/leetcode-solutions-python", split="train")
    except Exception as e:
        print(f"  Could not load LeetCode dataset: {e}")
        return [], {}

    functions = []
    skipped = {}

    for record in ds:
        # Field names vary — try common ones
        code = (
            record.get("python_solution", "")
            or record.get("solution", "")
            or record.get("code", "")
        ).strip()
        title = (
            record.get("title", "")
            or record.get("name", "")
            or record.get("problem_name", "")
        ).strip()
        description = (
            record.get("description", "")
            or record.get("content", "")
            or record.get("problem", "")
            or title
        ).strip()

        if not code or not description:
            skipped["missing_fields"] = skipped.get("missing_fields", 0) + 1
            continue

        # Strip HTML tags that appear in LeetCode descriptions
        description = re.sub(r"<[^>]+>", " ", description)
        description = re.sub(r"\s+", " ", description).strip()
        # Keep only the first sentence as docstring
        first_sentence = re.split(r"[.!?]", description)[0].strip()
        if not first_sentence:
            first_sentence = title
        docstring = first_sentence[:300]  # hard cap

        extracted = _extract_main_function(code)
        if not extracted:
            skipped["no_function_found"] = skipped.get("no_function_found", 0) + 1
            continue

        func_name = extracted["func_name"]
        func_code = extracted["code"]
        code_lines = extracted["code_lines"]

        keep, reason, quality = evaluate_corpus_quality(
            func_name, docstring, func_code, code_lines, eval_mode=False
        )
        if not keep:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        functions.append({
            "id": id_offset + len(functions),
            "func_name": func_name,
            "normalized_name": quality["normalized_name"],
            "code": func_code,
            "docstring": docstring,
            "code_lines": code_lines,
            "representation": quality["representation"],
            "repo": "leetcode/mhhmm",
            "path": f"leetcode/{title.replace(' ', '_').lower()}.py",
        })

    return functions, skipped


# ─── APPS helpers ──────────────────────────────────────────────

def _load_apps(id_offset: int, max_records: int = 5000) -> Tuple[List[Dict], Dict]:
    """Load APPS (competitive programming) dataset and return corpus-ready records."""
    from datasets import load_dataset

    print("\nLoading APPS dataset...")
    try:
        ds = load_dataset("codeparrot/apps", split="train")
    except Exception as e:
        print(f"  Could not load APPS dataset: {e}")
        return [], {}

    functions = []
    skipped = {}
    processed = 0

    for record in ds:
        if processed >= max_records:
            break
        processed += 1

        question = record.get("question", "").strip()
        solutions_raw = record.get("solutions", "")

        if not question or not solutions_raw:
            skipped["missing_fields"] = skipped.get("missing_fields", 0) + 1
            continue

        # solutions is a JSON string containing a list of solution strings
        try:
            solutions = json.loads(solutions_raw) if isinstance(solutions_raw, str) else solutions_raw
        except (json.JSONDecodeError, TypeError):
            skipped["bad_solutions_json"] = skipped.get("bad_solutions_json", 0) + 1
            continue

        if not solutions:
            skipped["no_solutions"] = skipped.get("no_solutions", 0) + 1
            continue

        # Try each solution until one yields a valid function
        extracted = None
        for sol in solutions[:3]:  # try up to 3 solutions
            extracted = _extract_main_function(sol)
            if extracted:
                break

        if not extracted:
            skipped["no_function_found"] = skipped.get("no_function_found", 0) + 1
            continue

        # Build docstring from question text (first sentence, no HTML)
        clean_q = re.sub(r"<[^>]+>", " ", question)
        clean_q = re.sub(r"\s+", " ", clean_q).strip()
        first_sentence = re.split(r"[.!?\n]", clean_q)[0].strip()
        docstring = first_sentence[:300] if first_sentence else clean_q[:300]

        func_name = extracted["func_name"]
        func_code = extracted["code"]
        code_lines = extracted["code_lines"]

        keep, reason, quality = evaluate_corpus_quality(
            func_name, docstring, func_code, code_lines, eval_mode=False
        )
        if not keep:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        functions.append({
            "id": id_offset + len(functions),
            "func_name": func_name,
            "normalized_name": quality["normalized_name"],
            "code": func_code,
            "docstring": docstring,
            "code_lines": code_lines,
            "representation": quality["representation"],
            "repo": "apps/codeparrot",
            "path": f"apps/problem_{record.get('problem_id', len(functions))}.py",
        })

    return functions, skipped


# ─── Main ─────────────────────────────────────────────────────

def download_and_prepare():
    from datasets import load_dataset

    # ── Step 1: CodeSearchNet ──────────────────────────────────
    print("=" * 60)
    print("STEP 1: CodeSearchNet Python Dataset")
    print("=" * 60)
    print("\nLoading from HuggingFace Hub...")
    csn_dataset = load_dataset(
        config.DATASET_NAME,
        config.DATASET_LANGUAGE,
    )
    for split, data in csn_dataset.items():
        print(f"  {split}: {len(data)} records")

    print("\nExtracting functions...")
    functions, eval_pairs, csn_skipped = _load_codesearchnet(csn_dataset)
    print(f"  Kept: {len(functions)}  |  Skipped: {sum(csn_skipped.values())}")
    for reason, count in sorted(csn_skipped.items()):
        print(f"    - {reason}: {count}")

    # ── Step 2: MBPP ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 2: MBPP (Basic Python Problems)")
    print("=" * 60)
    mbpp_functions, mbpp_skipped = _load_mbpp(id_offset=len(functions))
    print(f"  Kept: {len(mbpp_functions)}  |  Skipped: {sum(mbpp_skipped.values())}")
    for reason, count in sorted(mbpp_skipped.items()):
        print(f"    - {reason}: {count}")
    functions.extend(mbpp_functions)

    # ── Step 3: HumanEval ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 3: HumanEval")
    print("=" * 60)
    he_functions, he_skipped = _load_humaneval(id_offset=len(functions))
    print(f"  Kept: {len(he_functions)}  |  Skipped: {sum(he_skipped.values())}")
    for reason, count in sorted(he_skipped.items()):
        print(f"    - {reason}: {count}")
    functions.extend(he_functions)

    # ── Step 4: LeetCode ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 4: LeetCode Python Solutions")
    print("=" * 60)
    lc_functions, lc_skipped = _load_leetcode(id_offset=len(functions))
    print(f"  Kept: {len(lc_functions)}  |  Skipped: {sum(lc_skipped.values())}")
    for reason, count in sorted(lc_skipped.items()):
        print(f"    - {reason}: {count}")
    functions.extend(lc_functions)

    # ── Step 5: APPS ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 5: APPS (Competitive Programming)")
    print("=" * 60)
    apps_functions, apps_skipped = _load_apps(id_offset=len(functions))
    print(f"  Kept: {len(apps_functions)}  |  Skipped: {sum(apps_skipped.values())}")
    for reason, count in sorted(apps_skipped.items()):
        print(f"    - {reason}: {count}")
    functions.extend(apps_functions)

    # ── Step 6: Save ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 6: Saving")
    print("=" * 60)

    corpus_path = config.PROCESSED_DATA_DIR / "functions_corpus.pkl"
    with open(corpus_path, "wb") as f:
        pickle.dump(functions, f)
    print(f"Corpus saved: {corpus_path}")

    eval_path = config.PROCESSED_DATA_DIR / "eval_pairs.pkl"
    with open(eval_path, "wb") as f:
        pickle.dump(eval_pairs, f)
    print(f"Eval pairs saved: {eval_path}")

    sample_path = config.PROCESSED_DATA_DIR / "sample_functions.json"
    with open(sample_path, "w") as f:
        json.dump(functions[:10], f, indent=2)
    print(f"Sample saved: {sample_path}")

    # ── Summary ───────────────────────────────────────────────
    import numpy as np
    code_lengths = [f["code_lines"] for f in functions]
    doc_lengths  = [len(f["docstring"].split()) for f in functions]

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    csn_count  = len(functions) - len(mbpp_functions) - len(he_functions) - len(lc_functions) - len(apps_functions)
    print(f"  CodeSearchNet:  {csn_count}")
    print(f"  MBPP:           {len(mbpp_functions)}")
    print(f"  HumanEval:      {len(he_functions)}")
    print(f"  LeetCode:       {len(lc_functions)}")
    print(f"  APPS:           {len(apps_functions)}")
    print(f"  Total corpus:   {len(functions)}")
    print(f"  Eval pairs:     {len(eval_pairs)}")
    print(f"  Code lines  — mean: {np.mean(code_lengths):.1f}, "
          f"median: {np.median(code_lengths):.1f}")
    print(f"  Docstring words — mean: {np.mean(doc_lengths):.1f}, "
          f"median: {np.median(doc_lengths):.1f}")
    print(f"\nNext: python main_index.py")

    return functions, eval_pairs


if __name__ == "__main__":
    download_and_prepare()
