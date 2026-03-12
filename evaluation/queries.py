"""
Curated evaluation queries for Python code search.

Inspired by the Queries50 benchmark used in Ryu et al. (2025), 
adapted for Python-specific programming tasks.

Each query has:
  - query: natural language description
  - keywords: important terms for matching
  - category: type of programming task
"""

PYTHON_QUERIES_50 = [
    # ── List/Array Operations ──────────────────────────
    {
        "id": 1,
        "query": "sort a list of dictionaries by a specific key",
        "keywords": ["sort", "list", "dictionary", "key"],
        "category": "data_structures",
    },
    {
        "id": 2,
        "query": "flatten a nested list in python",
        "keywords": ["flatten", "nested", "list"],
        "category": "data_structures",
    },
    {
        "id": 3,
        "query": "remove duplicates from a list while preserving order",
        "keywords": ["remove", "duplicates", "list", "order"],
        "category": "data_structures",
    },
    {
        "id": 4,
        "query": "find the intersection of two lists",
        "keywords": ["intersection", "two", "lists"],
        "category": "data_structures",
    },
    {
        "id": 5,
        "query": "split a list into chunks of a given size",
        "keywords": ["split", "list", "chunks", "size"],
        "category": "data_structures",
    },
    
    # ── String Operations ──────────────────────────────
    {
        "id": 6,
        "query": "reverse a string in python",
        "keywords": ["reverse", "string"],
        "category": "strings",
    },
    {
        "id": 7,
        "query": "check if a string is a palindrome",
        "keywords": ["check", "string", "palindrome"],
        "category": "strings",
    },
    {
        "id": 8,
        "query": "convert a string to a datetime object",
        "keywords": ["convert", "string", "datetime"],
        "category": "strings",
    },
    {
        "id": 9,
        "query": "remove all whitespace from a string",
        "keywords": ["remove", "whitespace", "string"],
        "category": "strings",
    },
    {
        "id": 10,
        "query": "count the occurrences of each word in a string",
        "keywords": ["count", "occurrences", "word", "string"],
        "category": "strings",
    },
    
    # ── File I/O ───────────────────────────────────────
    {
        "id": 11,
        "query": "read a file line by line in python",
        "keywords": ["read", "file", "line"],
        "category": "file_io",
    },
    {
        "id": 12,
        "query": "write a list of strings to a file",
        "keywords": ["write", "list", "strings", "file"],
        "category": "file_io",
    },
    {
        "id": 13,
        "query": "read a json file into a dictionary",
        "keywords": ["read", "json", "file", "dictionary"],
        "category": "file_io",
    },
    {
        "id": 14,
        "query": "read a csv file into a list of dictionaries",
        "keywords": ["read", "csv", "file", "list", "dictionaries"],
        "category": "file_io",
    },
    {
        "id": 15,
        "query": "check if a file exists before opening it",
        "keywords": ["check", "file", "exists"],
        "category": "file_io",
    },
    
    # ── Dictionary Operations ──────────────────────────
    {
        "id": 16,
        "query": "merge two dictionaries in python",
        "keywords": ["merge", "two", "dictionaries"],
        "category": "data_structures",
    },
    {
        "id": 17,
        "query": "sort a dictionary by its values",
        "keywords": ["sort", "dictionary", "values"],
        "category": "data_structures",
    },
    {
        "id": 18,
        "query": "invert a dictionary swapping keys and values",
        "keywords": ["invert", "dictionary", "keys", "values"],
        "category": "data_structures",
    },
    {
        "id": 19,
        "query": "get the key with the maximum value in a dictionary",
        "keywords": ["key", "maximum", "value", "dictionary"],
        "category": "data_structures",
    },
    {
        "id": 20,
        "query": "filter a dictionary by a condition on values",
        "keywords": ["filter", "dictionary", "condition", "values"],
        "category": "data_structures",
    },
    
    # ── Math/Numbers ───────────────────────────────────
    {
        "id": 21,
        "query": "generate a random integer in a specific range",
        "keywords": ["generate", "random", "integer", "range"],
        "category": "math",
    },
    {
        "id": 22,
        "query": "check if a number is prime",
        "keywords": ["check", "number", "prime"],
        "category": "math",
    },
    {
        "id": 23,
        "query": "calculate the factorial of a number",
        "keywords": ["calculate", "factorial", "number"],
        "category": "math",
    },
    {
        "id": 24,
        "query": "compute the greatest common divisor of two numbers",
        "keywords": ["greatest", "common", "divisor", "gcd"],
        "category": "math",
    },
    {
        "id": 25,
        "query": "round a floating point number to n decimal places",
        "keywords": ["round", "float", "decimal", "places"],
        "category": "math",
    },
    
    # ── Date/Time ──────────────────────────────────────
    {
        "id": 26,
        "query": "get the current date and time in python",
        "keywords": ["current", "date", "time"],
        "category": "datetime",
    },
    {
        "id": 27,
        "query": "calculate the difference between two dates",
        "keywords": ["difference", "between", "dates"],
        "category": "datetime",
    },
    {
        "id": 28,
        "query": "convert a timestamp to a human readable date string",
        "keywords": ["convert", "timestamp", "date", "string"],
        "category": "datetime",
    },
    
    # ── HTTP/Web ───────────────────────────────────────
    {
        "id": 29,
        "query": "send an http get request and parse the json response",
        "keywords": ["http", "get", "request", "json", "response"],
        "category": "web",
    },
    {
        "id": 30,
        "query": "download a file from a url",
        "keywords": ["download", "file", "url"],
        "category": "web",
    },
    
    # ── Data Processing ────────────────────────────────
    {
        "id": 31,
        "query": "convert a list of tuples to a dictionary",
        "keywords": ["convert", "list", "tuples", "dictionary"],
        "category": "data_processing",
    },
    {
        "id": 32,
        "query": "group a list of items by a specific attribute",
        "keywords": ["group", "list", "items", "attribute"],
        "category": "data_processing",
    },
    {
        "id": 33,
        "query": "compute the moving average of a list of numbers",
        "keywords": ["moving", "average", "list", "numbers"],
        "category": "data_processing",
    },
    {
        "id": 34,
        "query": "normalize a list of numbers to a range between 0 and 1",
        "keywords": ["normalize", "list", "numbers", "range"],
        "category": "data_processing",
    },
    
    # ── Error Handling / Utilities ─────────────────────
    {
        "id": 35,
        "query": "retry a function call on failure with exponential backoff",
        "keywords": ["retry", "function", "failure", "backoff"],
        "category": "utilities",
    },
    {
        "id": 36,
        "query": "measure the execution time of a function",
        "keywords": ["measure", "execution", "time", "function"],
        "category": "utilities",
    },
    {
        "id": 37,
        "query": "create a decorator that caches function results",
        "keywords": ["decorator", "cache", "function", "results"],
        "category": "utilities",
    },
    {
        "id": 38,
        "query": "validate an email address using a regular expression",
        "keywords": ["validate", "email", "regular", "expression"],
        "category": "utilities",
    },
    {
        "id": 39,
        "query": "generate a random alphanumeric string of a given length",
        "keywords": ["generate", "random", "alphanumeric", "string"],
        "category": "utilities",
    },
    {
        "id": 40,
        "query": "convert a python object to a json string",
        "keywords": ["convert", "python", "object", "json", "string"],
        "category": "utilities",
    },
    
    # ── Advanced / Algorithms ──────────────────────────
    {
        "id": 41,
        "query": "implement binary search on a sorted list",
        "keywords": ["binary", "search", "sorted", "list"],
        "category": "algorithms",
    },
    {
        "id": 42,
        "query": "find all permutations of a list",
        "keywords": ["permutations", "list"],
        "category": "algorithms",
    },
    {
        "id": 43,
        "query": "implement a depth first search traversal of a graph",
        "keywords": ["depth", "first", "search", "graph"],
        "category": "algorithms",
    },
    {
        "id": 44,
        "query": "find the longest common subsequence of two strings",
        "keywords": ["longest", "common", "subsequence", "strings"],
        "category": "algorithms",
    },
    {
        "id": 45,
        "query": "compute the hash of a file using md5",
        "keywords": ["hash", "file", "md5"],
        "category": "algorithms",
    },
    
    # ── Classes / OOP ──────────────────────────────────
    {
        "id": 46,
        "query": "implement a singleton pattern in python",
        "keywords": ["singleton", "pattern", "class"],
        "category": "oop",
    },
    {
        "id": 47,
        "query": "create an iterator class that generates fibonacci numbers",
        "keywords": ["iterator", "class", "fibonacci", "numbers"],
        "category": "oop",
    },
    {
        "id": 48,
        "query": "deep copy a nested python object",
        "keywords": ["deep", "copy", "nested", "object"],
        "category": "utilities",
    },
    {
        "id": 49,
        "query": "convert a class instance to a dictionary",
        "keywords": ["convert", "class", "instance", "dictionary"],
        "category": "oop",
    },
    {
        "id": 50,
        "query": "create a context manager for database connections",
        "keywords": ["context", "manager", "database", "connections"],
        "category": "oop",
    },
]


def get_queries(category: str = None) -> list:
    """Get queries, optionally filtered by category."""
    if category:
        return [q for q in PYTHON_QUERIES_50 if q["category"] == category]
    return PYTHON_QUERIES_50


def get_query_texts() -> list:
    """Get just the query strings."""
    return [q["query"] for q in PYTHON_QUERIES_50]
