"""
indexer.py
----------
Unified indexing pipeline for the Hardware Test Codebase Agent.

Runs two passes over the codebase and stores everything in one ChromaDB collection:

  Pass 1 — Code chunks (chunk_type: "code")
    - Parses every .py file with tree-sitter
    - Extracts function/class blocks
    - Applies hybrid chunking (function boundaries + 80-line cap + sliding window)
    - Generates LLM summaries via Claude Haiku
    - Embeds summary + raw code together

  Pass 2 — Symbol chunks (chunk_type: "symbol")
    - Extracts intra-file symbol relationships (no LLM needed, no extra cost)
    - File-level: imports, top-level definitions, class method lists
    - Function-level: what each function calls, its parameters, decorators
    - Stored as structured text, embedded directly

Symbol chunks surface when users ask relationship questions like:
  "what does Run() call?"
  "what does ChannelCard import?"
  "what methods does CCTestClass have?"

Usage:
    uv run indexer.py --codebase_dir /path/to/your/codebase

Requirements:
    pip install tree-sitter==0.21.3 tree-sitter-python==0.21.0 anthropic chromadb openai flashrank
"""

import os
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import anthropic
import chromadb
from chromadb.utils import embedding_functions

from tree_sitter import Language, Parser
import tree_sitter_python as tspython


# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "codebase"
MAX_CHUNK_LINES  = 80
OVERLAP_LINES    = 10
EMBEDDING_MODEL  = "text-embedding-3-small"
SUMMARY_MODEL    = "claude-haiku-4-5-20251001"
SKIP_IF_INDEXED  = True    # Set False to force full re-index

EXCLUDE_DIRS = {"xlsxwriter", "site-packages", "venv", ".venv","__pycache__",".vs",".idea"}


# ── Tree-sitter Setup ─────────────────────────────────────────────────────────

def build_parser() -> Parser:
    PY_LANGUAGE = Language(tspython.language(), "python")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser


# ═════════════════════════════════════════════════════════════════════════════
# PASS 1 — CODE CHUNKS
# ═════════════════════════════════════════════════════════════════════════════

def extract_blocks_from_file(
    file_path: Path,
    codebase_root: Path,
    parser: Parser
) -> list[dict]:
    """
    Parse a Python file with tree-sitter and extract all function and class
    definitions as raw text blocks with line numbers.
    Fault-tolerant: tree-sitter parses around syntax errors automatically.
    """
    source_bytes  = file_path.read_bytes()
    source_text   = source_bytes.decode("utf-8", errors="replace")
    source_lines  = source_text.splitlines()
    relative_path = str(file_path.relative_to(codebase_root))

    tree   = parser.parse(source_bytes)
    blocks = []

    def walk(node):
        if node.type in ("function_definition", "async_function_definition", "class_definition"):
            name = "unknown"
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8", errors="replace")
                    break

            if node.type == "class_definition":
                name = f"class:{name}"

            line_start = node.start_point[0]
            line_end   = node.end_point[0] + 1
            raw_code   = "\n".join(source_lines[line_start:line_end])

            blocks.append({
                "file_path":     relative_path,
                "function_name": name,
                "line_start":    line_start + 1,
                "line_end":      line_end,
                "raw_code":      raw_code,
            })

        for child in node.children:
            walk(child)

    walk(tree.root_node)

    if not blocks:
        blocks.append({
            "file_path":     relative_path,
            "function_name": "module-level",
            "line_start":    1,
            "line_end":      len(source_lines),
            "raw_code":      source_text,
        })
    else:
        module_code = _extract_module_level(tree, source_lines)
        if module_code.strip():
            blocks.append({
                "file_path":     relative_path,
                "function_name": "module-level",
                "line_start":    1,
                "line_end":      len(source_lines),
                "raw_code":      module_code,
            })

    return blocks


def _extract_module_level(tree, source_lines: list[str]) -> str:
    """Extract lines that sit outside any function or class definition."""
    defined_lines = set()

    def mark(node):
        if node.type in ("function_definition", "async_function_definition", "class_definition"):
            for i in range(node.start_point[0], node.end_point[0] + 1):
                defined_lines.add(i)
        else:
            for child in node.children:
                mark(child)

    mark(tree.root_node)
    return "\n".join(
        line for i, line in enumerate(source_lines)
        if i not in defined_lines
    )


def split_into_chunks(blocks: list[dict]) -> list[dict]:
    """
    Hybrid chunking:
    - Blocks within MAX_CHUNK_LINES → kept as-is
    - Larger blocks → sliding window sub-chunks with OVERLAP_LINES overlap
    - Blocks over 4000 lines → skipped (auto-generated/minified code)
    """
    chunks = []

    for block in blocks:
        code_lines = block["raw_code"].splitlines()
        num_lines  = len(code_lines)

        if num_lines > 4000:
            print(f"  [skip] {block['file_path']}:{block['function_name']} exceeds 4000 lines")
            continue

        if num_lines <= MAX_CHUNK_LINES:
            chunks.append(block)
        else:
            sub_index = 0
            start     = 0

            while start < num_lines:
                end      = min(start + MAX_CHUNK_LINES, num_lines)
                sub_code = "\n".join(code_lines[start:end])

                chunks.append({
                    "file_path":     block["file_path"],
                    "function_name": f"{block['function_name']}__part{sub_index}",
                    "line_start":    block["line_start"] + start,
                    "line_end":      block["line_start"] + end - 1,
                    "raw_code":      sub_code,
                })

                sub_index  += 1
                next_start  = end - OVERLAP_LINES
                if next_start <= start:
                    next_start = start + MAX_CHUNK_LINES
                start = next_start

    print(f"[indexer] Produced {len(chunks)} code chunks after splitting")
    return chunks


def generate_summary(client: anthropic.Anthropic, chunk: dict) -> str:
    """Generate a plain-English summary for a code chunk via Claude Haiku."""
    prompt = f"""You are analyzing a Python code chunk from a hardware instrument testing codebase.
Write a concise 2-3 sentence plain-English summary of what this code does.
Focus on: what hardware or instrument it interacts with, what it computes or validates, and what it returns or produces.
Be specific. Do not say "this function does X" — just describe what it does directly.

File: {chunk['file_path']}
Function/Section: {chunk['function_name']}

Code:
```python
{chunk['raw_code']}
```

Summary:"""

    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def generate_summaries_for_chunks(chunks: list[dict], client: anthropic.Anthropic) -> list[dict]:
    """Generate LLM summaries for all code chunks with progress logging."""
    total = len(chunks)
    print(f"[indexer] Generating summaries for {total} code chunks...")

    for i, chunk in enumerate(chunks):
        try:
            chunk["summary"] = generate_summary(client, chunk)
        except Exception as e:
            print(f"  [warn] Summary failed for {chunk['file_path']}:{chunk['function_name']}: {e}")
            chunk["summary"] = f"Code from {chunk['file_path']} in section {chunk['function_name']}"

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] summaries generated...")

    print(f"[indexer] All summaries complete.")
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# PASS 2 — SYMBOL CHUNKS
# ═════════════════════════════════════════════════════════════════════════════

def _extract_function_calls(node) -> list[str]:
    """Recursively extract all unique function/method call names from a node."""
    calls = []

    def walk(n):
        if n.type == "call":
            func_node = n.child_by_field_name("function")
            if func_node:
                if func_node.type == "identifier":
                    calls.append(func_node.text.decode("utf-8", errors="replace"))
                elif func_node.type == "attribute":
                    attr = func_node.child_by_field_name("attribute")
                    obj  = func_node.child_by_field_name("object")
                    if attr and obj:
                        calls.append(
                            f"{obj.text.decode('utf-8', errors='replace')}"
                            f".{attr.text.decode('utf-8', errors='replace')}"
                        )
        for child in n.children:
            walk(child)

    walk(node)
    seen = set()
    return [c for c in calls if not (c in seen or seen.add(c))]


def _extract_parameters(node) -> list[str]:
    """Extract parameter names from a function definition node."""
    params      = []
    params_node = node.child_by_field_name("parameters")
    if not params_node:
        return params

    for child in params_node.children:
        if child.type == "identifier":
            name = child.text.decode("utf-8", errors="replace")
            if name not in ("self", "cls"):
                params.append(name)
        elif child.type in ("typed_parameter", "default_parameter", "typed_default_parameter"):
            for subchild in child.children:
                if subchild.type == "identifier":
                    name = subchild.text.decode("utf-8", errors="replace")
                    if name not in ("self", "cls"):
                        params.append(name)
                    break

    return params


def _extract_decorators(node) -> list[str]:
    """Extract decorator names from a function or class definition node."""
    decorators = []
    for child in node.children:
        if child.type == "decorator":
            for subchild in child.children:
                if subchild.type == "identifier":
                    decorators.append(subchild.text.decode("utf-8", errors="replace"))
                    break
                elif subchild.type == "attribute":
                    decorators.append(subchild.text.decode("utf-8", errors="replace"))
                    break
    return decorators


def _extract_imports(tree) -> list[str]:
    """Extract all import statements from a file's syntax tree."""
    imports = []

    def walk(node):
        if node.type in ("import_statement", "import_from_statement"):
            imports.append(node.text.decode("utf-8", errors="replace").strip())
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return imports


def extract_symbol_chunks_from_file(
    file_path: Path,
    codebase_root: Path,
    parser: Parser
) -> list[dict]:
    """
    Extract symbol relationship chunks from a Python file.

    Produces:
      - One file-level chunk: imports, top-level defines, class method lists
      - One chunk per function/method with meaningful call relationships
    """
    source_bytes  = file_path.read_bytes()
    source_text   = source_bytes.decode("utf-8", errors="replace")
    source_lines  = source_text.splitlines()
    relative_path = str(file_path.relative_to(codebase_root))

    tree    = parser.parse(source_bytes)
    chunks  = []

    # ── File-level symbols ─────────────────────────────────────────────────
    imports       = _extract_imports(tree)
    defines       = []
    class_methods = {}

    # ── Per-function symbols ───────────────────────────────────────────────
    func_symbol_chunks = []

    def walk(node, current_class: str = ""):
        if node.type == "class_definition":
            class_name = "unknown"
            for child in node.children:
                if child.type == "identifier":
                    class_name = child.text.decode("utf-8", errors="replace")
                    break
            defines.append(f"class:{class_name}")
            class_methods[class_name] = []
            for child in node.children:
                walk(child, current_class=class_name)

        elif node.type in ("function_definition", "async_function_definition"):
            func_name = "unknown"
            for child in node.children:
                if child.type == "identifier":
                    func_name = child.text.decode("utf-8", errors="replace")
                    break

            if current_class:
                class_methods.setdefault(current_class, []).append(func_name)
            else:
                defines.append(func_name)

            params     = _extract_parameters(node)
            calls      = _extract_function_calls(node)
            decorators = _extract_decorators(node)
            line_start = node.start_point[0] + 1
            line_end   = node.end_point[0] + 1

            # Only create a symbol chunk if there's meaningful relationship data
            if calls or params:
                context = f"in class {current_class}" if current_class else "at module level"
                lines   = [
                    f"FUNCTION SYMBOL: {func_name} ({context})",
                    f"FILE: {relative_path} (lines {line_start}–{line_end})",
                    "",
                ]
                if decorators:
                    lines.append(f"DECORATORS: {', '.join(decorators)}")
                if params:
                    lines.append(f"PARAMETERS: {', '.join(params)}")
                if calls:
                    lines.append(f"CALLS ({len(calls)} unique):")
                    for call in calls[:30]:
                        lines.append(f"  - {call}")

                document = "\n".join(lines)
                if current_class:
                    summary = (
                        f"{current_class}.{func_name}() in {relative_path} "
                        f"accepts ({', '.join(params) or 'no params'}) "
                        f"and calls: {', '.join(calls[:5]) or 'nothing'}."
                    )
                else:
                    summary = (
                        f"Function {func_name}() in {relative_path} "
                        f"accepts ({', '.join(params) or 'no params'}) "
                        f"and calls: {', '.join(calls[:5]) or 'nothing'}."
                    )

                chunk_id = hashlib.md5(
                    f"funcsym::{relative_path}::{func_name}::{line_start}".encode()
                ).hexdigest()

                func_symbol_chunks.append({
                    "chunk_id":      chunk_id,
                    "document":      f"SUMMARY: {summary}\n\nSYMBOL DETAIL:\n{document}",
                    "file_path":     relative_path,
                    "function_name": f"symbol::{func_name}",
                    "line_start":    line_start,
                    "line_end":      line_end,
                    "summary":       summary,
                    "chunk_type":    "symbol",
                })

            for child in node.children:
                walk(child, current_class=current_class)

        else:
            for child in node.children:
                walk(child, current_class=current_class)

    walk(tree.root_node)

    # ── File-level chunk ───────────────────────────────────────────────────
    file_lines = [f"FILE SYMBOL MAP: {relative_path}\n"]
    if imports:
        file_lines.append("IMPORTS:")
        for imp in imports:
            file_lines.append(f"  {imp}")
        file_lines.append("")
    if defines:
        file_lines.append("DEFINES (top-level):")
        for d in defines:
            file_lines.append(f"  {d}")
        file_lines.append("")
    if class_methods:
        file_lines.append("CLASS METHODS:")
        for class_name, methods in class_methods.items():
            file_lines.append(f"  {class_name}: {', '.join(methods)}")

    file_summary = (
        f"File {relative_path} imports {len(imports)} modules and defines "
        f"{len(defines)} top-level symbols. "
        f"Classes: {', '.join(class_methods.keys()) or 'none'}."
    )
    file_chunk_id = hashlib.md5(f"filesym::{relative_path}".encode()).hexdigest()

    chunks.append({
        "chunk_id":      file_chunk_id,
        "document":      f"SUMMARY: {file_summary}\n\nSYMBOLS:\n" + "\n".join(file_lines),
        "file_path":     relative_path,
        "function_name": "file-symbol-map",
        "line_start":    0,
        "line_end":      0,
        "summary":       file_summary,
        "chunk_type":    "symbol",
    })

    chunks.extend(func_symbol_chunks)
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# CHROMADB STORAGE
# ═════════════════════════════════════════════════════════════════════════════

def build_chunk_id(file_path: str, function_name: str, line_start: int) -> str:
    raw = f"{file_path}::{function_name}::{line_start}"
    return hashlib.md5(raw.encode()).hexdigest()


def store_all_chunks(
    code_chunks: list[dict],
    symbol_chunks: list[dict],
    openai_api_key: str
) -> None:
    """
    Store both code and symbol chunks in ChromaDB in one operation.
    Wipes and recreates the collection for a clean index.
    """
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[indexer] Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # ── Store code chunks ──────────────────────────────────────────────────
    print(f"[indexer] Storing {len(code_chunks)} code chunks...")
    _batch_insert(collection, [
        {
            "id":       build_chunk_id(c["file_path"], c["function_name"], c["line_start"]),
            "document": f"SUMMARY: {c['summary']}\n\nCODE:\n{c['raw_code']}",
            "metadata": {
                "file_path":     c["file_path"],
                "function_name": c["function_name"],
                "line_start":    c["line_start"],
                "line_end":      c["line_end"],
                "summary":       c["summary"],
                "chunk_type":    "code",
            }
        }
        for c in code_chunks
    ])

    # ── Store symbol chunks ────────────────────────────────────────────────
    print(f"[indexer] Storing {len(symbol_chunks)} symbol chunks...")
    _batch_insert(collection, [
        {
            "id":       c["chunk_id"],
            "document": c["document"],
            "metadata": {
                "file_path":     c["file_path"],
                "function_name": c["function_name"],
                "line_start":    c["line_start"],
                "line_end":      c["line_end"],
                "summary":       c["summary"],
                "chunk_type":    c["chunk_type"],
            }
        }
        for c in symbol_chunks
    ])

    total = collection.count()
    print(f"\n[indexer] ✓ Done. {total} total chunks stored in ChromaDB at '{CHROMA_DB_PATH}'")
    print(f"          Code chunks:   {len(code_chunks)}")
    print(f"          Symbol chunks: {len(symbol_chunks)}")


def _batch_insert(collection, items: list[dict], batch_size: int = 100):
    """Insert items into ChromaDB in batches."""
    total = len(items)
    for start in range(0, total, batch_size):
        batch = items[start: start + batch_size]
        collection.add(
            ids       = [i["id"]       for i in batch],
            documents = [i["document"] for i in batch],
            metadatas = [i["metadata"] for i in batch],
        )
        print(f"  [stored] {min(start + batch_size, total)}/{total}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_indexing_pipeline(codebase_dir: str, anthropic_api_key: str, openai_api_key: str):
    """Full pipeline: parse → code chunks → symbol chunks → summarize → store."""

    # ── Skip check ────────────────────────────────────────────────────────
    if SKIP_IF_INDEXED:
        chroma_path = Path(CHROMA_DB_PATH)
        if chroma_path.exists() and any(chroma_path.iterdir()):
            print(f"[indexer] ChromaDB already exists at '{CHROMA_DB_PATH}'. Skipping.")
            print(f"[indexer] Set SKIP_IF_INDEXED = False to force a full re-index.")
            return

    print(f"\n{'='*60}")
    print(f"  Hardware Test Codebase Indexer")
    print(f"  Codebase: {codebase_dir}")
    print(f"{'='*60}\n")

    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    codebase_root    = Path(codebase_dir)
    parser           = build_parser()

    python_files = [
        f for f in codebase_root.rglob("*.py")
        if not any(part in EXCLUDE_DIRS for part in f.parts)
    ]
    print(f"[indexer] Found {len(python_files)} Python files")

    # ── Pass 1: Code chunks ────────────────────────────────────────────────
    print(f"\n[indexer] Pass 1 — extracting code chunks...")
    all_blocks = []
    for file_path in python_files:
        blocks = extract_blocks_from_file(file_path, codebase_root, parser)
        all_blocks.extend(blocks)
    print(f"[indexer] Extracted {len(all_blocks)} function/class blocks")

    code_chunks = split_into_chunks(all_blocks)
    code_chunks = generate_summaries_for_chunks(code_chunks, anthropic_client)

    # ── Pass 2: Symbol chunks ──────────────────────────────────────────────
    print(f"\n[indexer] Pass 2 — extracting symbol chunks (no LLM cost)...")
    symbol_chunks = []
    for file_path in python_files:
        try:
            chunks = extract_symbol_chunks_from_file(file_path, codebase_root, parser)
            symbol_chunks.extend(chunks)
        except Exception as e:
            print(f"  [warn] Symbol extraction failed for {file_path}: {e}")
    print(f"[indexer] Extracted {len(symbol_chunks)} symbol chunks")

    # ── Store everything ───────────────────────────────────────────────────
    print(f"\n[indexer] Storing all chunks in ChromaDB...")
    store_all_chunks(code_chunks, symbol_chunks, openai_api_key)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Index a Python codebase for the HW Test Agent")
    arg_parser.add_argument("--codebase_dir", required=True, help="Path to codebase root")
    args = arg_parser.parse_args()

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key    = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key:
        raise ValueError("Missing ANTHROPIC_API_KEY")
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY")

    run_indexing_pipeline(args.codebase_dir, anthropic_key, openai_key)
