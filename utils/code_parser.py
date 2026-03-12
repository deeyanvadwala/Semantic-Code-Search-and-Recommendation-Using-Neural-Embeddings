"""
AST-based Python function extractor.

Used when indexing raw Python source files (e.g., from GitHub repos).
For CodeSearchNet data, functions are already extracted, but this module
is useful for indexing your own codebases.
"""

import ast
import os
from pathlib import Path
from typing import List, Dict, Optional


class PythonFunctionExtractor:
    """Extract functions and their metadata from Python source files using AST."""
    
    def __init__(self, min_lines: int = 2, require_docstring: bool = False):
        """
        Args:
            min_lines: Minimum number of lines for a function to be included
            require_docstring: If True, only include functions with docstrings
        """
        self.min_lines = min_lines
        self.require_docstring = require_docstring
    
    def extract_from_file(self, filepath: str) -> List[Dict]:
        """Extract all functions from a single Python file."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
            return self.extract_from_source(source, filepath)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return []
    
    def extract_from_source(self, source: str, filepath: str = "<unknown>") -> List[Dict]:
        """Extract all functions from Python source code string."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._extract_function_info(node, source, filepath)
                if func_info:
                    functions.append(func_info)
        
        return functions
    
    def _extract_function_info(
        self, node: ast.FunctionDef, source: str, filepath: str
    ) -> Optional[Dict]:
        """Extract metadata from a single function AST node."""
        
        # Get source lines
        source_lines = source.split("\n")
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else self._estimate_end(node, source_lines)
        
        func_code = "\n".join(source_lines[start_line:end_line])
        code_lines = end_line - start_line
        
        # Apply minimum lines filter
        if code_lines < self.min_lines:
            return None
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        # Apply docstring requirement
        if self.require_docstring and not docstring:
            return None
        
        # Extract function signature
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            annotation = ""
            if arg.annotation:
                try:
                    annotation = ast.unparse(arg.annotation)
                except:
                    annotation = ""
            args.append({"name": arg_name, "type": annotation})
        
        # Extract return type
        return_type = ""
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except:
                return_type = ""
        
        # Extract decorators
        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except:
                pass
        
        return {
            "func_name": node.name,
            "code": func_code,
            "docstring": docstring,
            "code_lines": code_lines,
            "args": args,
            "return_type": return_type,
            "decorators": decorators,
            "filepath": filepath,
            "start_line": start_line + 1,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }
    
    def _estimate_end(self, node, source_lines):
        """Estimate function end line for older Python versions."""
        # Simple heuristic: find next line at same or lower indentation
        start = node.lineno - 1
        if start >= len(source_lines):
            return start + 1
        
        base_indent = len(source_lines[start]) - len(source_lines[start].lstrip())
        
        for i in range(start + 1, len(source_lines)):
            line = source_lines[i]
            if line.strip() == "":
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent:
                return i
        
        return len(source_lines)
    
    def extract_from_directory(self, directory: str, recursive: bool = True) -> List[Dict]:
        """Extract functions from all Python files in a directory."""
        directory = Path(directory)
        pattern = "**/*.py" if recursive else "*.py"
        
        all_functions = []
        for py_file in directory.glob(pattern):
            # Skip common non-source directories
            parts = py_file.parts
            skip_dirs = {"venv", "env", ".venv", "__pycache__", "node_modules", ".git", "dist", "build"}
            if any(d in parts for d in skip_dirs):
                continue
            
            functions = self.extract_from_file(str(py_file))
            all_functions.extend(functions)
        
        print(f"Extracted {len(all_functions)} functions from {directory}")
        return all_functions


# ─── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    extractor = PythonFunctionExtractor(min_lines=2)
    
    test_code = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[-1] + dp[-2])
    return dp[n]

def add(a, b):
    return a + b
'''
    
    functions = extractor.extract_from_source(test_code, "test.py")
    for f in functions:
        print(f"Function: {f['func_name']}")
        print(f"  Lines: {f['code_lines']}")
        print(f"  Docstring: {f['docstring'][:50]}...")
        print()
