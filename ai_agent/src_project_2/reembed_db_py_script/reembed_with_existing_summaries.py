import ast
import re
import hashlib
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from tree_sitter import Language, Parser
import tree_sitter_python as tspython


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "codebase"

MAX_CHUNK_LINES = 80
OVERLAP_LINES = 10
EMBEDDING_MODEL = "text-embedding-3-small"

EXCLUDE_DIRS = {"xlsxwriter", "site-packages", "venv", ".venv"}

# Change this to your codebase root
CODEBASE_DIR = r"suite_loc_full" #change


# ── ID compatibility with your original indexer ──────────────────────────────

def build_chunk_id(file_path: str, function_name: str, line_start: int) -> str:
    raw = f"{file_path}::{function_name}::{line_start}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Tree-sitter setup ─────────────────────────────────────────────────────────

def build_parser() -> Parser:
    py_language = Language(tspython.language(), "python")
    parser = Parser()
    parser.set_language(py_language)
    return parser


# ── Metadata sanitization for Chroma ──────────────────────────────────────────

def compact_metadata(meta: dict) -> dict:
    """
    Chroma rejects:
      - None values
      - empty lists
    Keep False/0/"" because those can be meaningful.
    """
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        cleaned[k] = v
    return cleaned


# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_identifier_text(node) -> str | None:
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace")
    return None


def extract_calls_returns_asserts(code: str) -> tuple[list[str], bool, bool]:
    """
    Heuristic static analysis using Python ast.
    Returns:
      - calls: list of function/method call names
      - returns_value: whether any return statement returns a value
      - assertions_present: whether assert/assert_* appears
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        import textwrap
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            return [], False, False

    calls = []
    returns_value = False
    assertions_present = False

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            nonlocal assertions_present
            name = None

            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr

            if name:
                calls.append(name)
                if name.startswith("assert"):
                    assertions_present = True

            self.generic_visit(node)

        def visit_Return(self, node):
            nonlocal returns_value
            if node.value is not None:
                returns_value = True
            self.generic_visit(node)

        def visit_Assert(self, node):
            nonlocal assertions_present
            assertions_present = True
            self.generic_visit(node)

    Visitor().visit(tree)

    seen = set()
    deduped = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return deduped, returns_value, assertions_present


def detect_instrument_mentions(code: str) -> list[str]:
    """
    Heuristic keyword detection. Tune these patterns for your environment.
    """
    text = code.lower()

    patterns = {
        "power supply": [
            r"\bpsu\b", r"power\s*supply", r"e36\d+", r"n67\d+"
        ],
        "DMM": [
            r"\bdmm\b", r"digital multimeter", r"3446\d", r"keithley"
        ],
        "oscilloscope": [
            r"\bscope\b", r"oscilloscope", r"mso", r"dso"
        ],
        "signal generator": [
            r"signal generator", r"function generator", r"\bawg\b"
        ],
        "electronic load": [
            r"electronic load", r"\beload\b", r"load current"
        ],
        "DAQ": [
            r"\bdaq\b", r"data acquisition"
        ],
        "serial": [
            r"\bserial\b", r"\buart\b", r"com\d+", r"pyserial"
        ],
        "visa": [
            r"\bvisa\b", r"pyvisa", r"resource manager", r"gpib", r"\busb::"
        ],
        "I2C": [
            r"\bi2c\b"
        ],
        "SPI": [
            r"\bspi\b"
        ],
        "GPIO": [
            r"\bgpio\b"
        ],
    }

    found = []
    for label, regs in patterns.items():
        if any(re.search(p, text) for p in regs):
            found.append(label)

    return found


def make_fallback_summary(chunk: dict) -> str:
    kind = chunk["chunk_type"]
    symbol = chunk["symbol"]
    calls = chunk.get("calls", [])
    file_path = chunk["file_path"]

    if calls:
        calls_text = ", ".join(calls[:5])
        return f"{kind} {symbol} from {file_path}; calls {calls_text}."
    return f"{kind} {symbol} from {file_path}."


# ── Parsing / extraction ──────────────────────────────────────────────────────

def extract_module_level(tree, source_lines: list[str]) -> str:
    defined_lines = set()

    def mark(node):
        if node.type in ("function_definition", "async_function_definition", "class_definition"):
            for i in range(node.start_point[0], node.end_point[0] + 1):
                defined_lines.add(i)
        else:
            for child in node.children:
                mark(child)

    mark(tree.root_node)

    module_lines = [
        line for i, line in enumerate(source_lines)
        if i not in defined_lines
    ]
    return "\n".join(module_lines)


def extract_blocks_from_file(file_path: Path, codebase_root: Path, parser: Parser) -> list[dict]:
    """
    IMPORTANT:
    - function_name stays compatible with your old indexer so chunk IDs match
    - symbol / parent_class / chunk_type are richer metadata fields
    """
    source_bytes = file_path.read_bytes()
    source_text = source_bytes.decode("utf-8", errors="replace")
    source_lines = source_text.splitlines()
    relative_path = str(file_path.relative_to(codebase_root))

    tree = parser.parse(source_bytes)
    blocks = []

    def walk(node, parent_class: str | None = None):
        if node.type in ("function_definition", "async_function_definition", "class_definition"):
            name = safe_identifier_text(node) or "unknown"

            line_start = node.start_point[0]
            line_end = node.end_point[0] + 1
            raw_code = "\n".join(source_lines[line_start:line_end])

            if node.type == "class_definition":
                old_function_name = f"class:{name}"
                symbol = name
                chunk_type = "class"
                parent_class_name = None
                next_parent = name
            else:
                old_function_name = name
                symbol = f"{parent_class}.{name}" if parent_class else name
                chunk_type = "method" if parent_class else "function"
                parent_class_name = parent_class
                next_parent = parent_class

            calls, returns_value, assertions_present = extract_calls_returns_asserts(raw_code)
            instrument_mentions = detect_instrument_mentions(raw_code)

            blocks.append({
                "file_path": relative_path,
                "function_name": old_function_name,
                "line_start": line_start + 1,
                "line_end": line_end,
                "raw_code": raw_code,
                "symbol": symbol,
                "chunk_type": chunk_type,
                "parent_class": parent_class_name,
                "instrument_mentions": instrument_mentions,
                "calls": calls,
                "returns_value": returns_value,
                "assertions_present": assertions_present,
            })

            for child in node.children:
                walk(child, next_parent)
            return

        for child in node.children:
            walk(child, parent_class)

    walk(tree.root_node)

    if not blocks:
        raw_code = source_text
        calls, returns_value, assertions_present = extract_calls_returns_asserts(raw_code)
        instrument_mentions = detect_instrument_mentions(raw_code)

        blocks.append({
            "file_path": relative_path,
            "function_name": "module-level",
            "line_start": 1,
            "line_end": len(source_lines),
            "raw_code": raw_code,
            "symbol": relative_path,
            "chunk_type": "module",
            "parent_class": None,
            "instrument_mentions": instrument_mentions,
            "calls": calls,
            "returns_value": returns_value,
            "assertions_present": assertions_present,
        })
    else:
        module_code = extract_module_level(tree, source_lines)
        if module_code.strip():
            calls, returns_value, assertions_present = extract_calls_returns_asserts(module_code)
            instrument_mentions = detect_instrument_mentions(module_code)

            blocks.append({
                "file_path": relative_path,
                "function_name": "module-level",
                "line_start": 1,
                "line_end": len(source_lines),
                "raw_code": module_code,
                "symbol": relative_path,
                "chunk_type": "module",
                "parent_class": None,
                "instrument_mentions": instrument_mentions,
                "calls": calls,
                "returns_value": returns_value,
                "assertions_present": assertions_present,
            })

    return blocks


def split_into_chunks(blocks: list[dict]) -> list[dict]:
    """
    Preserves your old chunk naming scheme for oversized blocks.
    """
    chunks = []

    for block in blocks:
        max_block_lines = 4000
        code_lines = block["raw_code"].splitlines()
        num_lines = len(code_lines)

        if num_lines > max_block_lines:
            print(f"  [skip] {block['file_path']}:{block['function_name']} exceeds {max_block_lines} lines")
            continue

        if num_lines <= MAX_CHUNK_LINES:
            chunks.append(block)
        else:
            sub_index = 0
            start = 0

            while start < num_lines:
                end = min(start + MAX_CHUNK_LINES, num_lines)
                sub_code = "\n".join(code_lines[start:end])

                calls, returns_value, assertions_present = extract_calls_returns_asserts(sub_code)
                instrument_mentions = detect_instrument_mentions(sub_code)

                chunks.append({
                    "file_path": block["file_path"],
                    "function_name": f"{block['function_name']}__part{sub_index}",
                    "line_start": block["line_start"] + start,
                    "line_end": block["line_start"] + end - 1,
                    "raw_code": sub_code,
                    "symbol": f"{block['symbol']}__part{sub_index}",
                    "chunk_type": block["chunk_type"],
                    "parent_class": block["parent_class"],
                    "instrument_mentions": instrument_mentions,
                    "calls": calls,
                    "returns_value": returns_value,
                    "assertions_present": assertions_present,
                })

                sub_index += 1
                next_start = end - OVERLAP_LINES
                if next_start <= start:
                    next_start = start + MAX_CHUNK_LINES
                start = next_start

    print(f"[reembed] Produced {len(chunks)} chunks after compatible splitting")
    return chunks


# ── Existing summary loader ───────────────────────────────────────────────────

def load_existing_summaries() -> dict[str, str]:
    """
    Attempts to load summaries from the existing collection.
    If the collection does not exist, returns an empty dict.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"[reembed] No existing collection '{COLLECTION_NAME}' found. Will use fallback summaries.")
        return {}

    offset = 0
    limit = 200
    summaries = {}

    while True:
        batch = collection.get(
            limit=limit,
            offset=offset,
            include=["metadatas"],
        )

        ids = batch.get("ids", [])
        metadatas = batch.get("metadatas", [])

        if not ids:
            break

        for chunk_id, meta in zip(ids, metadatas):
            meta = meta or {}
            summaries[chunk_id] = meta.get("summary", "")

        offset += limit

    print(f"[reembed] Loaded {len(summaries)} existing summaries from Chroma")
    return summaries


# ── Rebuild chunks from codebase ──────────────────────────────────────────────

def rebuild_chunks_from_codebase(codebase_dir: str) -> list[dict]:
    parser = build_parser()
    codebase_root = Path(codebase_dir)

    python_files = [
        f for f in codebase_root.rglob("*.py")
        if not any(part in EXCLUDE_DIRS for part in f.parts)
    ]
    print(f"[reembed] Found {len(python_files)} Python files")

    all_blocks = []
    for file_path in python_files:
        all_blocks.extend(extract_blocks_from_file(file_path, codebase_root, parser))

    print(f"[reembed] Extracted {len(all_blocks)} blocks")
    chunks = split_into_chunks(all_blocks)
    return chunks


# ── Temp collection helpers ───────────────────────────────────────────────────

def create_temp_collection(client, embedding_fn):
    temp_name = f"{COLLECTION_NAME}_rebuild"

    try:
        client.delete_collection(temp_name)
        print(f"[reembed] Deleted stale temp collection '{temp_name}'")
    except Exception:
        pass

    temp_collection = client.create_collection(
        name=temp_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return temp_name, temp_collection


def replace_main_collection_from_temp(client, temp_name: str):
    temp_collection = client.get_collection(name=temp_name)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[reembed] Deleted old collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    final_collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    offset = 0
    limit = 200
    copied = 0

    while True:
        batch = temp_collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        ids = batch.get("ids", [])
        if not ids:
            break

        final_collection.add(
            ids=ids,
            documents=batch["documents"],
            metadatas=batch["metadatas"],
            embeddings=batch["embeddings"],
        )

        copied += len(ids)
        print(f"[reembed] Copied {copied} records into '{COLLECTION_NAME}'")
        offset += limit

    client.delete_collection(temp_name)
    print(f"[reembed] Deleted temp collection '{temp_name}'")


# ── Re-embed with old summaries + new metadata ───────────────────────────────

def recreate_collection_with_reembedding(chunks: list[dict], old_summaries: dict[str, str], openai_api_key: str):
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    temp_name, temp_collection = create_temp_collection(client, embedding_fn)

    ids = []
    documents = []
    metadatas = []

    missing_summary_count = 0

    for chunk in chunks:
        chunk_id = build_chunk_id(
            chunk["file_path"],
            chunk["function_name"],
            chunk["line_start"]
        )

        summary = old_summaries.get(chunk_id, "")
        if not summary:
            missing_summary_count += 1
            summary = make_fallback_summary(chunk)

        document = f"SUMMARY: {summary}\n\nCODE:\n{chunk['raw_code']}"

        meta = compact_metadata({
            "file_path": chunk["file_path"],
            "function_name": chunk["function_name"],
            "line_start": chunk["line_start"],
            "line_end": chunk["line_end"],
            "summary": summary,
            "symbol": chunk["symbol"],
            "chunk_type": chunk["chunk_type"],
            "parent_class": chunk["parent_class"],
            "instrument_mentions": chunk["instrument_mentions"],
            "calls": chunk["calls"],
            "returns_value": chunk["returns_value"],
            "assertions_present": chunk["assertions_present"],
        })

        ids.append(chunk_id)
        documents.append(document)
        metadatas.append(meta)

    batch_size = 100
    total = len(ids)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        temp_collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"[reembed] Indexed {end}/{total} into temp collection")

    replace_main_collection_from_temp(client, temp_name)

    print("\n[reembed] Done")
    print(f"[reembed] Total chunks re-embedded: {total}")
    print(f"[reembed] Chunks missing an old summary: {missing_summary_count}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable")

    old_summaries = load_existing_summaries()
    chunks = rebuild_chunks_from_codebase(CODEBASE_DIR)
    recreate_collection_with_reembedding(chunks, old_summaries, openai_key)


if __name__ == "__main__":
    main()