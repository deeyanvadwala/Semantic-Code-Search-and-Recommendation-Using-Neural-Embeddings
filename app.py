"""
Streamlit UI for Semantic Code Search.
Run with:  streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import config
from models.semantic_search import SemanticSearchEngine
from utils.code_parser import PythonFunctionExtractor

# ─── Constants ───────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.015
FALLBACK = "We are currently working on it — will get back to you soon."
MAX_RECENT = 8

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Code Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ════════════════════════════════════════════
   RESET & PAGE
════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: "Inter", -apple-system, sans-serif;
    /* Claude: warm off-white */
    background-color: #F5F0E8;
    color: #1A1A1A;
}

/* Force Claude warm-white everywhere */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main, section.main {
    background-color: #F5F0E8 !important;
}
[data-testid="stHeader"] {
    background-color: #F5F0E8 !important;
    border-bottom: none !important;
}

/* Narrow centered column */
.block-container {
    max-width: 720px !important;
    padding: 0 1.5rem 5rem 1.5rem !important;
    margin: 0 auto !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }

/* ════════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    /* Claude: dark brown-black sidebar */
    background-color: #1C1917 !important;
    border-right: 1px solid #2C2825 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.25rem; }

/* All sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #CCCCCC !important;
}
section[data-testid="stSidebar"] .stDivider,
section[data-testid="stSidebar"] hr {
    border-color: #2A2A2A !important;
}

/* Sidebar ghost buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #BBBBBB !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 0.75rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: background 0.12s !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
    background: #2F2F2F !important;
    color: #FFFFFF !important;
    border: 1px solid #3A3A3A !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #3A3A3A !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed #3A3A3A !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #888888 !important;
}

/* ════════════════════════════════════════════
   MAIN CONTENT TEXT
════════════════════════════════════════════ */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] div,
[data-testid="stCaptionContainer"],
[data-testid="stText"] {
    color: #1A1A1A !important;
}

/* ════════════════════════════════════════════
   GREETING
════════════════════════════════════════════ */
.greeting {
    text-align: center;
    padding: 5rem 0 2.5rem 0;
}
.greeting-title {
    font-size: 2rem;
    font-weight: 500;
    color: #1A1A1A;
    letter-spacing: -0.02em;
}

/* ════════════════════════════════════════════
   SEARCH BAR — unified pill (input + buttons)
════════════════════════════════════════════ */
/* Outer pill */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"]) {
    background: #FFFFFF !important;
    border: 1.5px solid #DDD5C8 !important;
    border-radius: 999px !important;
    padding: 6px 6px 6px 0 !important;
    gap: 4px !important;
    align-items: center !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08) !important;
    overflow: hidden !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"]):focus-within {
    border-color: #D4783A !important;
    box-shadow: 0 0 0 3px rgba(212,120,58,0.12) !important;
}
/* Column wrappers inside pill — kill their own background */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    > div[data-testid="stColumn"] > div {
    background: #FFFFFF !important;
    padding: 0 !important;
}
/* Input — white, no own border */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stTextInput"] {
    margin-bottom: 0 !important;
}
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stTextInput"] > div,
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stTextInput"] > div > div {
    background: #FFFFFF !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stTextInput"] input {
    font-family: "Inter", sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 0.5rem 0.75rem 1.2rem !important;
    border-radius: 0 !important;
    border: none !important;
    background: #FFFFFF !important;
    color: #1A1A1A !important;
    box-shadow: none !important;
    outline: none !important;
}
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stTextInput"] input::placeholder {
    color: #A89A8A !important;
}
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stTextInput"] input:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
/* Clear × button — small grey circle */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stButton"]:not(:last-child) button {
    width: 30px !important;
    height: 30px !important;
    min-height: 30px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 0.72rem !important;
    background: #EDE7DF !important;
    color: #6A5A4A !important;
    border: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s !important;
}
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stButton"]:not(:last-child) button:hover {
    background: #D4783A !important;
    color: #FFFFFF !important;
}
/* Send ↑ button — orange filled circle */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stButton"]:last-child button {
    width: 38px !important;
    height: 38px !important;
    min-height: 38px !important;
    max-width: 38px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    background: #D4783A !important;
    color: #FFFFFF !important;
    border: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s !important;
    line-height: 1 !important;
}
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"])
    div[data-testid="stButton"]:last-child button:hover {
    background: #C46A2E !important;
}

/* ════════════════════════════════════════════
   BUTTONS (general)
════════════════════════════════════════════ */
/* Suggestion chips — Claude orange outline */
div[data-testid="stButton"] button[kind="secondary"] {
    background: transparent !important;
    border: 1.5px solid #DDD5C8 !important;
    border-radius: 999px !important;
    color: #5A4A3A !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    transition: all 0.12s !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: #D4783A !important;
    color: #D4783A !important;
    background: rgba(212,120,58,0.06) !important;
}
/* Primary buttons elsewhere */
div[data-testid="stButton"] button[kind="primary"] {
    background: #D4783A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.6rem !important;
    transition: background 0.15s !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #C46A2E !important;
}

/* ════════════════════════════════════════════
   SUGGESTION CHIPS
════════════════════════════════════════════ */
.sug-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 1.75rem;
}

/* ════════════════════════════════════════════
   RESULT CARD
════════════════════════════════════════════ */
.result-wrap { margin-top: 2rem; }

.result-card {
    /* Claude: white card on warm bg */
    background: #FFFFFF;
    border: 1px solid #E8DDD0;
    border-radius: 16px;
    padding: 20px 24px 16px 24px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.result-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.result-badge {
    background: #FFF0E6;
    color: #8C3E10;
    border: 1px solid #F5C9A8;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 99px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.lang-badge {
    background: #F0FDF4;
    color: #15803D;
    border: 1px solid #BBF7D0;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 99px;
}
.func-name {
    font-family: "JetBrains Mono", monospace;
    font-size: 1.05rem;
    font-weight: 500;
    color: #1A1A1A;
}
.result-meta {
    display: flex;
    gap: 14px;
    margin-bottom: 10px;
}
.meta-tag {
    font-size: 0.78rem;
    color: #7A6A5A;
}
.func-doc {
    font-size: 0.875rem;
    color: #4A3F35;
    line-height: 1.55;
    border-left: 2px solid #E8C9A8;
    padding-left: 12px;
    font-style: italic;
}

/* Code block */
[data-testid="stCode"] {
    border-radius: 12px !important;
    border: 1px solid #E8DDD0 !important;
    margin-top: 4px !important;
}

/* ════════════════════════════════════════════
   STATE CARDS  (warning / fallback)
════════════════════════════════════════════ */
.state-card {
    background: #FFFFFF;
    border: 1px solid #E8DDD0;
    border-radius: 14px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-top: 1.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.state-icon  { font-size: 1.8rem; margin-bottom: 0.6rem; display: block; }
.state-title { font-size: 0.95rem; font-weight: 500; color: #1A1A1A; margin-bottom: 0.3rem; }
.state-sub   { font-size: 0.85rem; color: #7A6A5A; }

.state-card.warn {
    border-left: 3px solid #D4783A;
    text-align: left;
    padding: 1.1rem 1.4rem;
}
.state-card.warn .state-title { color: #7C3910; }
.state-card.warn .state-sub   { color: #8C5230; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────

if "recent_queries"  not in st.session_state: st.session_state.recent_queries  = []
if "selected_query"  not in st.session_state: st.session_state.selected_query  = ""
if "input_version"   not in st.session_state: st.session_state.input_version   = 0

# ─── Engine ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading search engine…")
def load_engine() -> SemanticSearchEngine:
    engine = SemanticSearchEngine()
    engine.load()
    return engine

def _push_recent(q: str):
    r = st.session_state.recent_queries
    if q in r: r.remove(q)
    r.insert(0, q)
    st.session_state.recent_queries = r[:MAX_RECENT]

# ─── Query validator ──────────────────────────────────────────────────────────

_CASUAL = {
    "hello","hi","hey","yo","sup","bye","goodbye","thanks","thank","thx","ty",
    "why","when","where","who","ok","okay","alright","sure","yes","no","nope",
    "yeah","yep","please","sorry","lol","haha","good","bad","great","wow","hmm",
    "what","how","test","testing","help","stop","quit","exit","cool","nice",
}
_SIGNALS = {
    "list","dict","dictionary","array","tuple","set","queue","stack","tree",
    "graph","heap","linked","node","string","integer","int","float","number",
    "numbers","boolean","char","byte","sort","search","find","filter","map",
    "reduce","flatten","reverse","shuffle","merge","split","join","slice","chunk",
    "parse","convert","calculate","compute","check","validate","generate","create",
    "read","write","load","save","send","download","upload","fetch","connect",
    "query","insert","delete","update","remove","replace","count","sum","average",
    "mean","max","min","normalize","file","csv","json","xml","html","yaml","txt",
    "function","class","method","variable","loop","recursion","iterator",
    "generator","decorator","exception","error","thread","async","callback",
    "regex","pattern","factorial","fibonacci","prime","palindrome","anagram",
    "binary","hash","encrypt","compress","encode","decode","http","api","url",
    "database","sql","cache","index","python","lambda","dataframe","numpy",
    "pandas","swap","two","three","add","subtract","multiply","divide",
}

def _is_prog(query: str):
    words = query.lower().split()
    if not words:
        return False, "Please enter a query."
    if len(words) == 1 and words[0] in _CASUAL:
        return False, f'"{query}" is not a programming query.'
    if len(words) <= 3 and all(w in _CASUAL for w in words):
        return False, f'"{query}" doesn\'t look like a programming query.'
    if any(w in _SIGNALS for w in words):
        return True, ""
    if len(words) >= 4:
        return True, ""
    return False, f'"{query}" doesn\'t seem to be a programming query.'

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<p style='font-size:1rem;font-weight:600;color:#EEEEEE;"
        "padding:0 0.4rem;margin-bottom:1rem'>🔍 Code Search</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    if st.session_state.recent_queries:
        st.markdown(
            "<p style='font-size:0.68rem;font-weight:600;letter-spacing:0.09em;"
            "text-transform:uppercase;color:#555555;padding:0 0.4rem;"
            "margin-bottom:0.5rem'>Recent</p>",
            unsafe_allow_html=True,
        )
        for q in st.session_state.recent_queries:
            lbl = q if len(q) <= 30 else q[:27] + "…"
            if st.button(f"↩  {lbl}", key=f"rec_{q}", use_container_width=True):
                st.session_state.input_version += 1
                st.session_state.selected_query = q
                st.rerun()
        st.divider()

    st.markdown(
        "<p style='font-size:0.68rem;font-weight:600;letter-spacing:0.09em;"
        "text-transform:uppercase;color:#555555;padding:0 0.4rem;"
        "margin-bottom:0.5rem'>Index Your Code</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.8rem;color:#666666;padding:0 0.4rem;"
        "margin-bottom:0.75rem'>Upload a .py file to add your own functions.</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload", type=["py"], label_visibility="collapsed")
    if uploaded:
        if st.button("Add to Index", type="primary", use_container_width=True):
            engine = load_engine()
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="wb") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            funcs = PythonFunctionExtractor(min_lines=2).extract_from_file(tmp_path)
            os.unlink(tmp_path)
            if funcs:
                with st.spinner(f"Embedding {len(funcs)} functions…"):
                    added = engine.add_functions(funcs)
                st.success(f"Added {added} functions.")
            else:
                st.warning("No functions found.")

    st.divider()
    st.markdown(
        "<p style='font-size:0.75rem;color:#444444;line-height:1.8;"
        "padding:0 0.4rem'>"
        "Model: multilingual-e5-large<br>"
        "Index: FAISS · CodeSearchNet</p>",
        unsafe_allow_html=True,
    )

# ─── Greeting ────────────────────────────────────────────────────────────────

default_val = st.session_state.pop("selected_query", "") or ""
input_key   = f"main_query_{st.session_state.input_version}"
current_val = st.session_state.get(input_key, default_val)

if not current_val.strip():
    st.markdown("""
    <div class="greeting">
        <div class="greeting-title">What code are you looking for?</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)

# ─── Search bar (unified pill: input | × clear | ↑ send) ─────────────────────

if current_val.strip():
    col_input, col_clear, col_send = st.columns([8.5, 0.75, 0.75])
else:
    _ci, _cs = st.columns([9.25, 0.75])
    col_input, col_clear, col_send = _ci, None, _cs

with col_input:
    query_input = st.text_input(
        label="Search",
        value=default_val,
        placeholder="Ask anything…  e.g. sort a list of dictionaries by a key",
        label_visibility="collapsed",
        key=input_key,
    )

if col_clear is not None:
    with col_clear:
        if st.button("✕", key="clear_btn"):
            st.session_state.input_version += 1
            st.session_state.selected_query = ""
            st.rerun()

with col_send:
    search_btn = st.button("↑", type="primary")

query = query_input.strip()

# ─── Suggestions (landing only) ──────────────────────────────────────────────

if not query:
    suggestions = [
        "calculate factorial of a number",
        "check if a string is a palindrome",
        "sort a list of dictionaries by a key",
        "read a CSV file into a list",
        "find the longest common subsequence",
    ]
    for i, sug in enumerate(suggestions):
        if st.button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.input_version += 1
            st.session_state.selected_query = sug
            st.rerun()

# ─── Results ─────────────────────────────────────────────────────────────────

if (search_btn or query) and query:
    is_valid, warn_msg = _is_prog(query)

    if not is_valid:
        st.markdown(f"""
        <div class="state-card warn">
            <span class="state-icon">⚠️</span>
            <div class="state-title">{warn_msg}</div>
            <div class="state-sub">Try describing a Python function, e.g. "check if a number is prime".</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        engine = load_engine()
        _push_recent(query)

        with st.spinner("Searching…"):
            results = engine.search(query, top_k=1)

        if not results or results[0]["score"] < CONFIDENCE_THRESHOLD:
            st.markdown(f"""
            <div class="state-card">
                <span class="state-icon">🔧</span>
                <div class="state-title">{FALLBACK}</div>
                <div class="state-sub">Try rephrasing, or upload a .py file in the sidebar.</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            best     = results[0]
            lines    = best.get("code_lines") or len(best["code"].split("\n"))
            repo     = best.get("repo", "")
            doc      = best["docstring"].split("\n")[0].strip() if best.get("docstring") else ""
            repo_html = f'<span class="meta-tag">📦 {repo}</span>' if repo else ""
            doc_html  = f'<div class="func-doc">{doc}</div>' if doc else ""

            st.markdown(f"""
            <div class="result-wrap">
                <div class="result-card">
                    <div class="result-header">
                        <span class="result-badge">Best Match</span>
                        <span class="lang-badge">Python</span>
                        <span class="func-name">{best["func_name"]}</span>
                    </div>
                    <div class="result-meta">
                        <span class="meta-tag">📄 {lines} lines</span>
                        {repo_html}
                    </div>
                    {doc_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.code(best["code"], language="python")
