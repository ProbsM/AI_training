import chromadb

client  = chromadb.PersistentClient(path="./chroma_db")
col     = client.get_collection("codebase")
results = col.get(where={"chunk_type": "documentation"})

count = len(results["ids"])
print(f"Documentation chunks found: {count}")

for i, meta in enumerate(results["metadatas"]):
    print(f"  {i+1}. {meta['file_path']} — {meta['summary']}")