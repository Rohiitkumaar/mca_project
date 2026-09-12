import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import CHROMA_PATH, DOCUMENTS_DIR, get_embeddings

COLLECTION_NAME = "langchain"


def get_chroma_client() -> chromadb.ClientAPI:
    """Returns an active Chroma PersistentClient, clearing stale cache if stopped."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.get_version()  # Tests if Rust bindings are alive
        return client
    except Exception:
        SharedSystemClient.clear_system_cache()
        return chromadb.PersistentClient(path=CHROMA_PATH)


def get_vector_store() -> Chroma:
    """Initializes and returns the LangChain Chroma vector store."""
    client = get_chroma_client()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings()
    )


def list_indexed_documents() -> List[Dict[str, Any]]:
    """Returns a summary list of all indexed PDF documents in persistent storage."""
    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(COLLECTION_NAME)
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) or []
    except Exception:
        metadatas = []

    # Count chunks per document (normalized by basename)
    chunk_counts: Dict[str, int] = {}
    for meta in metadatas:
        if meta and "source" in meta:
            src = os.path.basename(meta["source"])
            chunk_counts[src] = chunk_counts.get(src, 0) + 1

    docs_info = []
    # Collect all unique sources from DB and DOCUMENTS_DIR
    all_sources = set(chunk_counts.keys())
    if os.path.exists(DOCUMENTS_DIR):
        for f in os.listdir(DOCUMENTS_DIR):
            if f.lower().endswith(".pdf"):
                all_sources.add(f)

    for filename in sorted(all_sources):
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        size_kb = 0.0
        modified_time = "Unknown"
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            size_kb = round(stat.st_size / 1024, 2)
            modified_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

        docs_info.append({
            "filename": filename,
            "chunks": chunk_counts.get(filename, 0),
            "size_kb": size_kb,
            "modified_time": modified_time,
            "exists_on_disk": os.path.exists(file_path)
        })

    return docs_info


def get_total_chunks() -> int:
    """Returns total chunks stored in the vector database."""
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


def index_pdf(file_bytes: bytes, filename: str) -> int:
    """Saves PDF to documents directory, splits text, tags with source metadata, and persists to ChromaDB."""
    clean_filename = os.path.basename(filename)
    file_path = os.path.join(DOCUMENTS_DIR, clean_filename)

    # 1. Save file to persistent storage
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # 2. If document was previously indexed, delete old chunks to avoid duplication
    client = get_chroma_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    try:
        collection.delete(where={"source": clean_filename})
        collection.delete(where={"source": f"./{clean_filename}"})
    except Exception:
        pass

    # 3. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    if not docs:
        return 0

    # 4. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 5. Attach consistent source metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = clean_filename
        chunk.metadata["filename"] = clean_filename
        chunk.metadata["chunk_id"] = i

    # 6. Add to Chroma vector database
    vector_db = get_vector_store()
    vector_db.add_documents(chunks)

    return len(chunks)


def delete_pdf(filename: str) -> bool:
    """Deletes all chunks belonging to a specific PDF and removes the file from disk."""
    clean_filename = os.path.basename(filename)

    # Delete from Chroma
    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(COLLECTION_NAME)
        collection.delete(where={"source": clean_filename})
        collection.delete(where={"source": f"./{clean_filename}"})
    except Exception:
        pass

    # Delete from disk
    file_path = os.path.join(DOCUMENTS_DIR, clean_filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    return True


def replace_pdf(filename: str, new_file_bytes: bytes) -> int:
    """Replaces an existing PDF by removing old chunks and re-indexing new content."""
    return index_pdf(new_file_bytes, filename)


def reset_all_documents() -> bool:
    """Deletes all collections, resets Chroma cache, and wipes the documents directory."""
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    SharedSystemClient.clear_system_cache()

    # Clear documents folder
    if os.path.exists(DOCUMENTS_DIR):
        for item in os.listdir(DOCUMENTS_DIR):
            p = os.path.join(DOCUMENTS_DIR, item)
            try:
                if os.path.isfile(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except Exception:
                pass

    return True
