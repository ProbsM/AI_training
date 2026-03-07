"""
retriever.py
------------
Retrieval layer for the Hardware Test Codebase Agent.

Given a natural language query, retrieves the most relevant code chunks
from ChromaDB using a three-phase approach:
  1. Query expansion  → generate alternative phrasings (rule-based, free)
  2. Vector search    → search all phrasings, merge + deduplicate results
  3. Re-ranking       → flashrank scores merged candidates, top K to LLM

Features:
  - Query expansion: converts natural language to likely code identifiers
    e.g. "dpin epa shmoo" → "DPIN_EPA_FreqShmoo", "DpinEpaShmoo", etc.
  - Dynamic chunk count: complex queries get 5 chunks, simple get 3
  - File-path filtering: "look only in ChannelCard" restricts retrieval

Usage (as a module):
    from retriever import CodebaseRetriever
    retriever = CodebaseRetriever()
    chunks = retriever.retrieve("how does the dpin epa shmoo test work?")
    chunks = retriever.retrieve("look only in ChannelCard — how does setup work?")
    chunks = retriever.retrieve("generate a new test script for voltage checking")
"""

import os
import re
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from flashrank import Ranker, RerankRequest


# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_PATH      = "./chroma_db"
COLLECTION_NAME     = "codebase"
EMBEDDING_MODEL     = "text-embedding-3-small"

VECTOR_SEARCH_TOP_K = 20   # Candidates per query expansion phrasing
SIMPLE_TOP_K        = 3    # Final chunks for simple queries
COMPLEX_TOP_K       = 5    # Final chunks for complex queries

# Keywords that trigger complex query mode (more chunks)
COMPLEX_KEYWORDS = {
    "generate", "create", "write", "build", "implement",
    "debug", "fix", "diagnose", "troubleshoot",
    "refactor", "modify", "update", "change", "edit",
    "explain in detail", "walk me through", "step by step",
}

# Patterns for file-path filtering
FILTER_PATTERNS = [
    r"(?:look\s+only\s+in|only\s+in|look\s+in|search\s+in|filter\s+to|within|inside)\s+([A-Za-z0-9_]+)",
]


# ── Query Expansion ───────────────────────────────────────────────────────────

def _to_snake_case(text: str) -> str:
    """'dpin epa shmoo' → 'dpin_epa_shmoo'"""
    return "_".join(text.strip().lower().split())


def _to_screaming_snake(text: str) -> str:
    """'dpin epa shmoo' → 'DPIN_EPA_SHMOO'"""
    return "_".join(text.strip().upper().split())


def _to_camel_case(text: str) -> str:
    """'dpin epa shmoo' → 'DpinEpaShmoo'"""
    return "".join(word.capitalize() for word in text.strip().split())


def _to_screaming_underscore_camel(text: str) -> str:
    """
    'dpin epa shmoo' → 'DPIN_EPA_Shmoo'
    Handles the common pattern in this codebase where module names mix
    SCREAMING prefix words with a CamelCase suffix.
    e.g. DPIN_EPA_FreqShmoo, HPCC_SpecCheck, HAL_InitCheck
    """
    words = text.strip().split()
    if len(words) <= 1:
        return text
    # First words screaming, last word capitalized
    return "_".join(w.upper() for w in words[:-1]) + "_" + words[-1].capitalize()


def _abbreviation_expansions(text: str) -> list[str]:
    """
    Expand known abbreviations common in hardware test codebases.
    Add more entries here as you discover patterns in your codebase.
    """
    expansions = {
        "dpin":  ["digital pin", "DPIN"],
        "epa":   ["enhanced performance architecture", "EPA"],
        "hpcc":  ["hp channel card", "HPCC"],
        "hal":   ["hardware abstraction layer", "HAL"],
        "dmm":   ["digital multimeter", "DMM"],
        "psu":   ["power supply", "PSU"],
        "tdr":   ["time domain reflectometry", "TDR"],
        "freq":  ["frequency", "Freq"],
        "shmoo": ["frequency sweep", "shmoo"],
        "spec":  ["specification", "Spec"],
        "cal":   ["calibration", "Cal"],
        "tiu":   ["test interface unit", "TIU"],
    }

    text_lower = text.lower()
    extra = []
    for abbr, full_forms in expansions.items():
        if abbr in text_lower.split():
            for form in full_forms:
                extra.append(text_lower.replace(abbr, form))
    return extra

STOP_WORDS = {"how", "does", "the", "what", "is", "a", "an", 
              "of", "to", "do", "work", "works", "tell", 
              "me", "about", "explain", "where", "when", "why"}

def expand_query(query: str) -> list[str]:
    """
    Generate alternative phrasings of a query to improve retrieval
    against code identifiers that use different naming conventions.

    For "dpin epa shmoo" generates:
      - "dpin epa shmoo"              (original)
      - "dpin_epa_shmoo"              (snake_case)
      - "DPIN_EPA_SHMOO"              (SCREAMING_SNAKE)
      - "DpinEpaShmoo"                (CamelCase)
      - "DPIN_EPA_Shmoo"              (Mixed — common in this codebase)
      - "digital pin enhanced performance architecture frequency sweep" (abbr expansion)
    """
    clean_query = re.sub(
        r"(?:only\s+)?(?:in|within|inside|look\s+in|search\s+in|filter\s+to)\s+[A-Za-z0-9_]+\s*[—\-–]?\s*",
        "", query, flags=re.IGNORECASE
    ).strip()

    # Filter stop words to isolate the identifier phrase
    words     = clean_query.split()
    id_words  = [w for w in words if w.lower() not in STOP_WORDS]
    id_phrase = " ".join(id_words[:4]) if id_words else clean_query
    
    expansions = [
        clean_query,                              # original (cleaned)
        _to_snake_case(id_phrase),                # snake_case
        _to_screaming_snake(id_phrase),           # SCREAMING_SNAKE
        _to_camel_case(id_phrase),                # CamelCase
        _to_screaming_underscore_camel(id_phrase),# SCREAMING_CamelCase (codebase pattern)
    ]

    # Add abbreviation expansions
    expansions.extend(_abbreviation_expansions(id_phrase))

    # Deduplicate while preserving order, skip empty strings
    seen   = set()
    result = []
    for e in expansions:
        e_clean = e.strip()
        if e_clean and e_clean not in seen:
            seen.add(e_clean)
            result.append(e_clean)

    return result


# ── Intent Detection ──────────────────────────────────────────────────────────

def detect_complexity(query: str) -> bool:
    """Returns True if the query needs more chunks (complex keywords detected)."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in COMPLEX_KEYWORDS)


def detect_path_filter(query: str) -> Optional[str]:
    """
    Detect if the user wants to restrict retrieval to a specific module.
    Returns the filter string if found, None otherwise.
    """
    for pattern in FILTER_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


# ── Retriever ─────────────────────────────────────────────────────────────────

class CodebaseRetriever:
    """
    Three-phase retriever:
      Phase 1 - Rule-based query expansion
      Phase 2 - ChromaDB vector search across all expanded phrasings
      Phase 3 - flashrank re-ranking of merged candidates
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")

        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=EMBEDDING_MODEL
        )

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection    = self.chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef
        )

        self.ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir="/tmp/flashrank"
        )

        print(f"[retriever] Loaded collection '{COLLECTION_NAME}' "
              f"({self.collection.count()} chunks indexed)")


    def retrieve(self, query: str) -> list[dict]:
        """
        Main retrieval method. Automatically:
        - Expands the query into multiple phrasings
        - Detects complexity → uses 3 or 5 chunks
        - Detects path filter → restricts to a specific module/directory
        - Merges and deduplicates results across all phrasings
        - Re-ranks with flashrank

        Returns top_k most relevant chunks as a list of dicts.
        """

        # ── Phase 1: Query expansion ──────────────────────────────────────
        expanded_queries = expand_query(query)
        is_complex       = detect_complexity(query)
        path_filter      = detect_path_filter(query)
        top_k            = COMPLEX_TOP_K if is_complex else SIMPLE_TOP_K

        if is_complex:
            print(f"[retriever] Complex query → fetching {top_k} chunks")
        if path_filter:
            print(f"[retriever] Path filter → restricting to '{path_filter}'")
        if len(expanded_queries) > 1:
            print(f"[retriever] Query expanded to {len(expanded_queries)} phrasings")

        # ── Phase 2: Vector search across all phrasings ───────────────────
        seen_ids   = set()
        candidates = []

        for phrasing in expanded_queries:
            query_kwargs = dict(
                query_texts=[phrasing],
                n_results=min(VECTOR_SEARCH_TOP_K, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )

            if path_filter:
                query_kwargs["where"] = {
                    "file_path": {"$contains": path_filter}
                }

            try:
                results = self.collection.query(**query_kwargs)
            except Exception:
                # Path filter returned no results — fall back to unfiltered
                query_kwargs.pop("where", None)
                try:
                    results = self.collection.query(**query_kwargs)
                except Exception:
                    continue

            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    candidates.append({
                        "id":           chunk_id,
                        "document":     results["documents"][0][i],
                        "metadata":     results["metadatas"][0][i],
                        "vector_score": 1 - results["distances"][0][i],
                    })

        if not candidates:
            return []

        # ── Phase 3: Re-ranking ────────────────────────────────────────────
        rerank_request = RerankRequest(
            query=query,   # Re-rank against the original query, not expansions
            passages=[{"id": c["id"], "text": c["document"]} for c in candidates]
        )
        reranked  = self.ranker.rerank(rerank_request)
        score_map = {r["id"]: r["score"] for r in reranked}

        for c in candidates:
            c["rerank_score"] = score_map.get(c["id"], 0.0)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_candidates = candidates[:top_k]

        # ── Format output ──────────────────────────────────────────────────
        chunks = []
        for c in top_candidates:
            meta     = c["metadata"]
            raw_code = c["document"].split("CODE:\n", 1)[-1] if "CODE:\n" in c["document"] else c["document"]

            chunks.append({
                "file_path":     meta.get("file_path", "unknown"),
                "function_name": meta.get("function_name", "unknown"),
                "line_start":    meta.get("line_start", 0),
                "line_end":      meta.get("line_end", 0),
                "summary":       meta.get("summary", ""),
                "raw_code":      raw_code,
                "score":         round(c["rerank_score"], 4),
            })

        return chunks


    def format_chunks_for_prompt(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a clean string for the LLM prompt."""
        if not chunks:
            return "No relevant code found in the codebase for this query."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"--- Chunk {i} ---\n"
                f"File:     {chunk['file_path']} (lines {chunk['line_start']}–{chunk['line_end']})\n"
                f"Function: {chunk['function_name']}\n"
                f"Summary:  {chunk['summary']}\n\n"
                f"```python\n{chunk['raw_code']}\n```"
            )

        return "\n\n".join(parts)


# ── CLI for quick testing ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "how does the dpin epa shmoo test work?"

    print(f"\nQuery: {query}")
    print(f"Expansions: {expand_query(query)}")
    print(f"{'='*60}")

    retriever = CodebaseRetriever()
    chunks    = retriever.retrieve(query)

    print(f"\nRetrieved {len(chunks)} chunks\n")
    print(retriever.format_chunks_for_prompt(chunks))
