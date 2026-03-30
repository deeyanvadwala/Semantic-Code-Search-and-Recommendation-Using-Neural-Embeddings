"""
Text preprocessing utilities for code and natural language queries.
"""

import re
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def preprocess_code(code: str) -> str:
    """
    Preprocess Python code for embedding.
    
    Keeps the code mostly intact but normalizes whitespace
    and removes excessive comments that don't add semantic value.
    """
    # Remove excessive blank lines (keep max 1)
    code = re.sub(r'\n{3,}', '\n\n', code)
    
    # Normalize whitespace within lines
    lines = code.split('\n')
    processed_lines = []
    for line in lines:
        # Keep indentation but normalize trailing whitespace
        processed_lines.append(line.rstrip())
    
    return '\n'.join(processed_lines).strip()


def preprocess_query(query: str) -> str:
    """
    Preprocess a natural language query for embedding.
    
    Normalizes the query text while preserving programming terms.
    """
    # Lowercase
    query = query.lower().strip()
    
    # Remove excessive whitespace
    query = re.sub(r'\s+', ' ', query)
    
    # Remove trailing punctuation that doesn't add meaning
    query = query.rstrip('?.!')
    
    return query


def extract_parameters(code: str) -> Dict[str, List[str]]:
    """
    Extract parameter names, type annotations, and return type from a function.

    Uses the AST so it works on any valid Python function without regex.

    Returns:
        Dict with keys:
            'params'      — list of readable "name type" strings (self/cls excluded)
            'return_type' — return annotation as a string, or empty string
    """
    import ast as _ast

    params = []
    return_type = ""

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return {"params": params, "return_type": return_type}

    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue

        skip = {"self", "cls"}
        for arg in node.args.args:
            name = arg.arg
            if name in skip:
                continue
            if arg.annotation:
                try:
                    ann = _ast.unparse(arg.annotation)
                except Exception:
                    ann = ""
                # Flatten generics: List[int] → list int
                ann = re.sub(r"[\[\],]", " ", ann).strip()
                ann = re.sub(r"\s+", " ", ann).lower()
                params.append(f"{name} {ann}".strip())
            else:
                params.append(name)

        # Return type
        if node.returns:
            try:
                rt = _ast.unparse(node.returns)
                rt = re.sub(r"[\[\],]", " ", rt).strip()
                return_type = re.sub(r"\s+", " ", rt).lower()
            except Exception:
                pass

        break  # only process the outermost function

    return {"params": params, "return_type": return_type}


def create_code_representation(func_name: str, docstring: str, code: str) -> str:
    """
    Create the text representation of a function for embedding.

    Combines function name, parameter names + types, return type, and
    docstring summary into a single natural language passage.

    Incorporating parameter names and return type directly addresses the
    progress report goal of richer code representation beyond docstrings alone.

    Args:
        func_name: The function name
        docstring: The function's docstring
        code: The function source code

    Returns:
        A natural language representation of the function
    """
    parts = []

    # 1. Readable function name (snake_case → words)
    readable_name = func_name.replace('_', ' ').strip()
    if readable_name:
        parts.append(readable_name)

    # 2. Parameter names and types extracted from the signature
    sig = extract_parameters(code)
    if sig["params"]:
        parts.append("takes " + " ".join(sig["params"]))
    if sig["return_type"]:
        parts.append("returns " + sig["return_type"])

    # 3. Docstring summary (primary semantic signal)
    if docstring:
        first_para = docstring.split('\n\n')[0].strip()
        first_para = re.sub(
            r'^(Parameters|Args|Returns|Raises|Example).*',
            '', first_para, flags=re.DOTALL
        ).strip()
        if first_para:
            parts.append(first_para)

    # 4. Fallback when no docstring: pull identifiers from the function body
    if not docstring:
        identifiers = extract_identifiers(code)
        if identifiers:
            parts.append(' '.join(identifiers[:20]))

    return ' '.join(parts) if parts else func_name


def extract_identifiers(code: str) -> List[str]:
    """Extract meaningful variable/function names from code."""
    # Remove strings and comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = re.sub(r'"[^"]*"', '', code)
    code = re.sub(r"'[^']*'", '', code)
    
    # Extract identifiers (variable names, function calls)
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    
    # Remove Python keywords and common tokens
    python_keywords = {
        'def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while',
        'try', 'except', 'finally', 'with', 'as', 'import', 'from',
        'in', 'not', 'and', 'or', 'is', 'None', 'True', 'False',
        'self', 'cls', 'pass', 'break', 'continue', 'raise', 'yield',
        'lambda', 'global', 'nonlocal', 'assert', 'del', 'async', 'await',
    }
    
    filtered = [tok for tok in identifiers if tok not in python_keywords and len(tok) > 1]
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for tok in filtered:
        if tok not in seen:
            seen.add(tok)
            unique.append(tok)
    
    return unique


def split_camel_case(name: str) -> str:
    """Convert camelCase or PascalCase to space-separated words."""
    # Insert space before uppercase letters
    result = re.sub(r'([A-Z])', r' \1', name)
    return result.lower().strip()


def normalize_function_name(name: str) -> str:
    """Normalize a function name to readable text."""
    # Handle snake_case
    name = name.replace('_', ' ')
    # Handle camelCase
    name = split_camel_case(name)
    # Clean up
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def evaluate_corpus_quality(
    func_name: str,
    docstring: str,
    code: str,
    code_lines: int,
    eval_mode: bool = False,
) -> Tuple[bool, str, Dict[str, str]]:
    """
    Decide whether a function should be kept in the search corpus.

    The corpus should favor functions that are descriptive and queryable.
    Generic method names and boilerplate docstrings tend to pollute retrieval.
    """
    min_doc_chars = (
        config.MIN_EVAL_DOCSTRING_CHARS if eval_mode else config.MIN_DOCSTRING_CHARS
    )
    min_code_lines = (
        config.MIN_EVAL_CODE_LINES if eval_mode else config.MIN_CODE_LINES
    )

    func_name = func_name.strip()
    docstring = docstring.strip()
    code = code.strip()
    normalized_name = normalize_function_name(func_name)
    representation = create_code_representation(func_name, docstring, code)

    metadata = {
        "normalized_name": normalized_name,
        "representation": representation,
    }

    if not func_name:
        return False, "missing_name", metadata
    if not docstring or len(docstring) < min_doc_chars:
        return False, "weak_docstring", metadata
    if code_lines < min_code_lines:
        return False, "too_short_code", metadata
    if code_lines > config.MAX_CODE_LINES:
        return False, "too_long_code", metadata
    if len(docstring.split()) > config.MAX_DOCSTRING_WORDS:
        return False, "docstring_too_long", metadata
    if _looks_like_boilerplate_docstring(docstring):
        return False, "boilerplate_docstring", metadata
    if _is_generic_function_name(func_name, normalized_name, docstring, code):
        return False, "generic_name", metadata
    if len(representation.split()) < config.MIN_REPRESENTATION_TOKENS:
        return False, "thin_representation", metadata

    return True, "kept", metadata


def _looks_like_boilerplate_docstring(docstring: str) -> bool:
    """Return True for docstrings that provide little retrieval value."""
    text = re.sub(r"\s+", " ", docstring.strip()).lower()
    if not text:
        return True

    boilerplate_phrases = (
        "todo",
        "tbd",
        "not implemented",
        "test function",
        "helper function",
        "internal helper",
        "see base class",
        "override default behavior",
    )
    if any(phrase in text for phrase in boilerplate_phrases):
        return True

    words = text.split()
    if len(words) < 3:
        return True

    non_alpha_words = sum(1 for w in words if not re.search(r"[a-z]", w))
    return non_alpha_words > len(words) // 2


def _is_generic_function_name(
    func_name: str,
    normalized_name: str,
    docstring: str,
    code: str,
) -> bool:
    """
    Detect low-signal function names that usually hurt natural-language search.

    We keep them if the docstring or code makes their purpose specific enough.
    """
    base_name = func_name.strip().split(".")[-1].strip("_").lower()
    if not base_name:
        return True
    if base_name not in config.GENERIC_FUNCTION_NAMES:
        return False

    doc_lower = docstring.lower()
    normalized_tokens = normalized_name.split()
    code_identifiers = extract_identifiers(code)[:12]

    # Keep generic names when there is enough extra signal in the docstring/code.
    # Thresholds are intentionally low: even "Add two numbers" (2 unique words)
    # or a function with 3 distinct identifiers is specific enough to be queryable.
    has_specific_name = len(normalized_tokens) >= 2
    has_specific_doc = len(set(doc_lower.split()) - {base_name, "the", "a", "an"}) >= 2
    has_specific_code = len(code_identifiers) >= 3
    return not (has_specific_name or has_specific_doc or has_specific_code)
