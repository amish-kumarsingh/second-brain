from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGManager:
    """Handles ingestion, retrieval, and querying of notes using ChromaDB + embeddings."""

    def __init__(self, db_path: str = ".chromadb", collection_name: str = "notes"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    # --- Helper ---
    def embed_text(self, text: str):
        """Generate embedding for given text using SentenceTransformer."""
        return self.model.encode(text).tolist()

    # --- Ingestion ---
    def ingest_folder(self, folder_path: str = "data/notes"):
        """Read all .txt files, split into chunks, and store them in ChromaDB."""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        for file_path in folder.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self.text_splitter.split_text(content)
            print(f"📄 Splitting {file_path.name} into {len(chunks)} chunks...")

            for i, chunk in enumerate(chunks):
                doc_id = f"{file_path.stem}_{i}"
                embedding = self.embed_text(chunk)

                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "filename": file_path.name,
                        "chunk_index": i
                    }],
                )

            print(f"✅ Ingested {file_path.name} ({len(chunks)} chunks)")

    # --- Reset ---
    def reset_collection(self):
        """Deletes the ChromaDB collection for a clean start."""
        try:
            self.client.delete_collection(self.collection.name)
            print(f"🧹 Collection '{self.collection.name}' deleted successfully.")
        except Exception as e:
            print(f"⚠️ Could not delete collection: {e}")

    # --- Query ---
    def query_notes(self, query: str, n_results: int = 3):
        """Search for relevant chunks."""
        query_embedding = self.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            print("❌ No matching results found.")
            return None

        seen = set()
        for doc, meta in zip(docs, metas):
            key = f"{meta['filename']}_{meta['chunk_index']}"
            if key in seen:
                continue
            seen.add(key)
            print(f"\n📘 {meta['filename']}:\n{doc[:400]}\n---")

        return results

    # --- RAG Retrieval ---
    def rag_retrieve(self, query: str, n_results: int = 3) -> str:
        """
        Retrieve top-n relevant chunks and return a clean RAG context block.
        """
        query_embedding = self.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        if not results["documents"] or not results["documents"][0]:
            return "No relevant information found in your knowledge base."

        context_blocks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            filename = meta.get("filename", "Unknown file")
            chunk_info = f"[Source: {filename}]"
            context_blocks.append(f"{chunk_info}\n{doc.strip()}\n")

        context = "\n---\n".join(context_blocks)
        return f"Here are some relevant notes from your knowledge base:\n\n{context}\n"
