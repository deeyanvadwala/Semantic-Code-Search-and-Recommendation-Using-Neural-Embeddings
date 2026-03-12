"""
Text preprocessing utilities for code and natural language queries.
"""

import re
from typing import List


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


def create_code_representation(func_name: str, docstring: str, code: str) -> str:
    """
    Create the text representation of a function for embedding.
    
    This combines the function name and docstring into a single
    natural language description. Following the approach from
    Ryu et al. (2025), we use the docstring as a proxy for the
    LLM-generated code summary (which requires GPU infrastructure
    beyond our scope).
    
    Args:
        func_name: The function name
        docstring: The function's docstring
        code: The function source code (used as fallback)
    
    Returns:
        A natural language representation of the function
    """
    parts = []
    
    # Parse function name: convert snake_case to readable text
    readable_name = func_name.replace('_', ' ').strip()
    if readable_name:
        parts.append(readable_name)
    
    # Add docstring (primary semantic signal)
    if docstring:
        # Take first paragraph of docstring (summary line)
        first_para = docstring.split('\n\n')[0].strip()
        # Clean up common docstring artifacts
        first_para = re.sub(r'^(Parameters|Args|Returns|Raises|Example).*', 
                           '', first_para, flags=re.DOTALL).strip()
        if first_para:
            parts.append(first_para)
    
    # Fallback: if no docstring, extract key tokens from code
    if not docstring:
        # Extract meaningful identifiers from code
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
