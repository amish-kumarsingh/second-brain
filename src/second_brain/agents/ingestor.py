from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from second_brain.utils import get_tracer
from opentelemetry.trace import Status, StatusCode
from typing import Any

tracer = get_tracer("second_brain.rag_manager")


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
        with tracer.start_as_current_span("rag_manager.ingest_folder") as span:
            span.set_attribute("folder_path", folder_path)
            folder = Path(folder_path)
            
            if not folder.exists():
                span.set_status(Status(StatusCode.ERROR, f"Folder not found: {folder_path}"))
                raise FileNotFoundError(f"Folder not found: {folder_path}")

            txt_files = list(folder.glob("*.txt"))
            span.set_attribute("total_files", len(txt_files))
            
            total_chunks = 0
            for file_path in txt_files:
                with tracer.start_as_current_span("ingest_file") as file_span:
                    file_span.set_attribute("filename", file_path.name)
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        file_span.set_attribute("file_size", len(content))
                        chunks = self.text_splitter.split_text(content)
                        file_span.set_attribute("chunks_count", len(chunks))
                        total_chunks += len(chunks)
                        
                        print(f"📄 Splitting {file_path.name} into {len(chunks)} chunks...")

                        for i, chunk in enumerate(chunks):
                            doc_id = f"{file_path.stem}_{i}"
                            with tracer.start_as_current_span("embed_and_store") as embed_span:
                                embedding = self.embed_text(chunk)
                                embed_span.set_attribute("embedding_dim", len(embedding))
                                
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
                        file_span.set_status(Status(StatusCode.OK))
                        
                    except Exception as e:
                        file_span.set_status(Status(StatusCode.ERROR, str(e)))
                        file_span.record_exception(e)
                        raise
            
            span.set_attribute("total_chunks_ingested", total_chunks)
            span.set_status(Status(StatusCode.OK))

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
        with tracer.start_as_current_span("rag_manager.query_notes") as span:
            span.set_attribute("query", query[:100])  # Limit length
            span.set_attribute("n_results", n_results)
            
            try:
                with tracer.start_as_current_span("embed_query") as embed_span:
                    query_embedding = self.embed_text(query)
                    embed_span.set_attribute("embedding_dim", len(query_embedding))

                with tracer.start_as_current_span("vector_search") as search_span:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results
                    )
                    
                    documents = results.get("documents")
                    metadatas = results.get("metadatas")
                    
                    docs: list[str] = []
                    metas: list[Any] = []
                    
                    if documents and len(documents) > 0 and documents[0]:
                        docs = documents[0]
                        search_span.set_attribute("results_found", len(docs))
                    else:
                        search_span.set_attribute("results_found", 0)
                        
                    if metadatas and len(metadatas) > 0 and metadatas[0]:
                        metas = list(metadatas[0])  # type: ignore[arg-type]  # Convert to list to match type

                    if not docs:
                        span.set_attribute("has_results", False)
                        span.set_status(Status(StatusCode.OK))
                        print("❌ No matching results found.")
                        return None

                    seen = set()
                    unique_files = set()
                    for doc, meta in zip(docs, metas):
                        key = f"{meta['filename']}_{meta['chunk_index']}"
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_files.add(meta.get('filename', 'Unknown'))
                        print(f"\n📘 {meta['filename']}:\n{doc[:400]}\n---")

                    span.set_attribute("has_results", True)
                    span.set_attribute("results_count", len(docs))
                    span.set_attribute("unique_files", len(unique_files))
                    span.set_status(Status(StatusCode.OK))
                    return results
                    
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    # --- RAG Retrieval ---
    def rag_retrieve(self, query: str, n_results: int = 3) -> str:
        """
        Retrieve top-n relevant chunks and return a clean RAG context block.
        """
        with tracer.start_as_current_span("rag_manager.rag_retrieve") as span:
            span.set_attribute("query", query[:100])  # Limit length
            span.set_attribute("n_results", n_results)
            
            try:
                with tracer.start_as_current_span("embed_query") as embed_span:
                    query_embedding = self.embed_text(query)
                    embed_span.set_attribute("embedding_dim", len(query_embedding))

                with tracer.start_as_current_span("vector_search") as search_span:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results
                    )
                    
                    documents_list = results.get("documents")
                    if documents_list and len(documents_list) > 0:
                        first_doc_list = documents_list[0]
                        if first_doc_list:
                            search_span.set_attribute("results_found", len(first_doc_list))
                        else:
                            search_span.set_attribute("results_found", 0)
                    else:
                        search_span.set_attribute("results_found", 0)

                documents_list = results.get("documents")
                metadatas_list = results.get("metadatas")
                
                if not documents_list or len(documents_list) == 0:
                    span.set_attribute("has_results", False)
                    span.set_status(Status(StatusCode.OK))
                    return "No relevant information found in your knowledge base."
                
                first_docs = documents_list[0]
                if not first_docs:
                    span.set_attribute("has_results", False)
                    span.set_status(Status(StatusCode.OK))
                    return "No relevant information found in your knowledge base."

                context_blocks = []
                docs_list = first_docs
                metas_list: list = []
                if metadatas_list and len(metadatas_list) > 0 and metadatas_list[0]:
                    metas_list = metadatas_list[0]
                
                for doc, meta in zip(docs_list, metas_list):
                    filename = meta.get("filename", "Unknown file")
                    chunk_info = f"[Source: {filename}]"
                    context_blocks.append(f"{chunk_info}\n{doc.strip()}\n")

                context = "\n---\n".join(context_blocks)
                result = f"Here are some relevant notes from your knowledge base:\n\n{context}\n"
                
                span.set_attribute("has_results", True)
                span.set_attribute("context_length", len(result))
                span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
