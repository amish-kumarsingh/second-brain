from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Global model and client setup ---
client = chromadb.PersistentClient(path=".chromadb")
collection = client.get_or_create_collection(name="notes")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Helper for embedding ---
def embed_text(text: str):
    return model.encode(text).tolist()

# --- Ingest logic with chunking ---
def ingest_folder(folder_path: str = "data/notes"):
    folder = Path(folder_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    for file_path in folder.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content)
        print(f"📄 Splitting {file_path.name} into {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            doc_id = f"{file_path.stem}_{i}"
            embedding = embed_text(chunk)

            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "filename": file_path.name,
                    "chunk_index": i
                }],
            )

        print(f"✅ Ingested {file_path.name} ({len(chunks)} chunks)")

# --- Reset collection ---
def reset_collection():
    """Deletes the ChromaDB collection for a clean start."""
    try:
        client.delete_collection("notes")
        print("🧹 Collection 'notes' deleted successfully.")
    except Exception as e:
        print(f"⚠️ No existing collection found or error deleting: {e}")

# --- Query logic ---
def query_notes(query: str, n_results: int = 3):
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        print("❌ No matching results found.")
        return

    seen = set()
    for doc, meta in zip(docs, metas):
        key = f"{meta['filename']}_{meta['chunk_index']}"
        if key in seen:
            continue
        seen.add(key)

        print(f"\n📘 {meta['filename']}:")
        print(doc[:400] + "\n---")

    return results
