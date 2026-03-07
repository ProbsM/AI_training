"""
add_files_todb.py
-----------
Appends documentation files (.docx or .txt) to the existing ChromaDB index
without wiping the existing Python code chunks.

Chunks documents by paragraph, generates LLM summaries for each chunk,
embeds them, and appends to the existing ChromaDB collection.

Usage:
    uv run add_files_todb.py --file /path/to/ChannelCard_Notes.docx
    uv run add_files_todb.py --file /path/to/ChannelCard_Overview.txt
    uv run add_files_todb.py --file /path/to/notes.docx --preview   # preview chunks without indexing
    
    uv run add_files_todb.py --file ChannelCard_Notes.docx --delete
"""

import os
import re
import hashlib
import argparse
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions


# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "codebase"
EMBEDDING_MODEL  = "text-embedding-3-small"
SUMMARY_MODEL    = "claude-haiku-4-5-20251001"
MIN_PARA_WORDS   = 5    # Skip paragraphs shorter than this (headings, blank lines, etc.)


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_docx(file_path: Path) -> list[dict]:
    """
    Extract paragraphs from a .docx file.
    Returns a list of dicts with text and metadata.
    Requires: pip install python-docx
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for .docx files. Run: pip install python-docx")

    doc        = Document(file_path)
    paragraphs = []
    para_index = 0

    for para in doc.paragraphs:
        text = para.text.strip()

        # Skip empty or very short paragraphs
        if len(text.split()) < MIN_PARA_WORDS:
            continue

        paragraphs.append({
            "file_path":  str(file_path.name),
            "source":     str(file_path),
            "chunk_type": "documentation",
            "section":    para.style.name,   # e.g. "Heading 1", "Normal"
            "para_index": para_index,
            "raw_text":   text,
        })
        para_index += 1

    return paragraphs


def parse_txt(file_path: Path) -> list[dict]:
    """
    Extract paragraphs from a .txt file.
    Splits on double newlines (blank lines between paragraphs).
    """
    content    = file_path.read_text(encoding="utf-8", errors="replace")
    raw_paras  = re.split(r"\n\s*\n", content)
    paragraphs = []
    para_index = 0

    for para in raw_paras:
        text = para.strip()

        # Skip empty or very short paragraphs
        if len(text.split()) < MIN_PARA_WORDS:
            continue

        paragraphs.append({
            "file_path":  str(file_path.name),
            "source":     str(file_path),
            "chunk_type": "documentation",
            "section":    "text",
            "para_index": para_index,
            "raw_text":   text,
        })
        para_index += 1

    return paragraphs


def parse_file(file_path: Path) -> list[dict]:
    """Route to the correct parser based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .docx, .txt")


# ── Summaries ─────────────────────────────────────────────────────────────────

def generate_summary(client: anthropic.Anthropic, chunk: dict) -> str:
    """Generate a plain-English summary of a documentation paragraph."""
    prompt = f"""You are analyzing a documentation paragraph about a hardware instrument testing codebase.
Write a concise 1-2 sentence summary of what this paragraph describes.
Focus on: what module, test, or concept it explains, and what the key information is.

File: {chunk['file_path']}
Section: {chunk['section']}

Text:
{chunk['raw_text']}

Summary:"""

    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def generate_summaries(chunks: list[dict], client: anthropic.Anthropic) -> list[dict]:
    """Generate summaries for all chunks with progress logging."""
    total = len(chunks)
    print(f"[add_docs] Generating summaries for {total} chunks...")

    for i, chunk in enumerate(chunks):
        try:
            chunk["summary"] = generate_summary(client, chunk)
        except Exception as e:
            print(f"  [warn] Summary failed for chunk {i}: {e}")
            chunk["summary"] = f"Documentation from {chunk['file_path']}, paragraph {chunk['para_index']}"

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{total}] summaries generated...")

    return chunks


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def build_chunk_id(file_path: str, para_index: int) -> str:
    """Generate a stable unique ID for a documentation chunk."""
    raw = f"doc::{file_path}::{para_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def append_to_chromadb(chunks: list[dict], openai_api_key: str):
    """
    Append documentation chunks to the existing ChromaDB collection.
    Uses get_or_create_collection — does NOT wipe existing Python code chunks.
    """
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # get_or_create — preserves existing chunks
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    existing_count = collection.count()
    print(f"[add_docs] Existing collection has {existing_count} chunks — appending {len(chunks)} new chunks")

    # Insert in batches of 100
    batch_size = 100
    total      = len(chunks)

    for batch_start in range(0, total, batch_size):
        batch     = chunks[batch_start: batch_start + batch_size]
        ids       = []
        documents = []
        metadatas = []

        for chunk in batch:
            chunk_id = build_chunk_id(chunk["file_path"], chunk["para_index"])
            document = f"SUMMARY: {chunk['summary']}\n\nDOCUMENTATION:\n{chunk['raw_text']}"

            ids.append(chunk_id)
            documents.append(document)
            metadatas.append({
                "file_path":   chunk["file_path"],
                "function_name": f"doc::{chunk['section']}::para{chunk['para_index']}",
                "line_start":  chunk["para_index"],
                "line_end":    chunk["para_index"],
                "summary":     chunk["summary"],
                "chunk_type":  "documentation",
                "source":      chunk["source"],
            })

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  [appended] {min(batch_start + batch_size, total)}/{total} chunks added")

    print(f"\n[add_docs] ✓ Done. Collection now has {collection.count()} total chunks.")


def delete_doc(file_name: str, openai_api_key: str):
    """
    Delete all chunks from a specific document by file name.
    Example: delete_doc("ChannelCard_Notes.docx")
             uv run add_docs.py --file ChannelCard_Notes.docx --delete
    """
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection    = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )

    # Find all chunks from this document
    results = collection.get(where={"file_path": file_name})

    if not results["ids"]:
        print(f"[add_docs] No chunks found for '{file_name}'")
        return

    collection.delete(ids=results["ids"])
    print(f"[add_docs] ✓ Deleted {len(results['ids'])} chunks from '{file_name}'")
    print(f"[add_docs] Collection now has {collection.count()} total chunks.")

# ── Preview ───────────────────────────────────────────────────────────────────

def preview_chunks(chunks: list[dict]):
    """Print a preview of parsed chunks without indexing."""
    print(f"\n[preview] Found {len(chunks)} paragraphs:\n")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk["raw_text"][:120].replace("\n", " ")
        print(f"  {i}. [{chunk['section']}] {preview}...")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(file_path: str, preview: bool = False, delete: bool = False):
    print(f"\n{'='*60}")
    print(f"  add_docs — Append documentation to ChromaDB")
    print(f"  File: {file_path}")
    print(f"{'='*60}\n")

    path   = Path(file_path)
    if delete:
        delete_doc(path.name, os.environ.get("OPENAI_API_KEY"))
        return
    chunks = parse_file(path)
    print(f"[add_docs] Parsed {len(chunks)} paragraphs from {path.name}")

    if not chunks:
        print("[add_docs] No paragraphs found. Check the file has content and paragraphs are longer than 10 words.")
        return

    if preview:
        preview_chunks(chunks)
        return

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key    = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key:
        raise ValueError("Missing ANTHROPIC_API_KEY")
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY")

    client = anthropic.Anthropic(api_key=anthropic_key)
    chunks = generate_summaries(chunks, client)
    append_to_chromadb(chunks, openai_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append documentation to ChromaDB")
    parser.add_argument("--file",    required=True, help="Path to .docx or .txt file")
    parser.add_argument("--preview", action="store_true", help="Preview chunks without indexing")
    parser.add_argument("--delete", action="store_true", help="Delete all chunks from the specified file")
    args = parser.parse_args()

    run(args.file, preview=args.preview, delete=args.delete)