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
import time

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


from langchain_core.documents import Document

def index_pdf(file_bytes: bytes, filename: str) -> int:
    """Saves PDF to documents directory, builds continuous cross-page text stream, splits with overlap, and persists to ChromaDB."""
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

    # 4. Construct continuous text stream with exact page boundary mapping
    full_text = ""
    page_spans = []  # (start_char_idx, end_char_idx, page_number)

    for i, page_doc in enumerate(docs):
        page_num = page_doc.metadata.get("page", i) + 1  # 1-indexed page number
        raw_page_content = page_doc.page_content or ""
        
        # Add clean page header tag to keep natural document structure
        start_idx = len(full_text)
        if full_text:
            full_text += "\n\n"
            start_idx = len(full_text)
        
        full_text += f"--- Page {page_num} ---\n{raw_page_content.strip()}"
        end_idx = len(full_text)
        page_spans.append((start_idx, end_idx, page_num))

    # 5. Split continuous text across page boundaries with generous overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
    )
    text_chunks = splitter.split_text(full_text)
    if not text_chunks:
        return 0

    # 6. Map each chunk back to its spanning page(s) and attach metadata
    search_cursor = 0
    chunks: List[Document] = []

    for i, chunk_text in enumerate(text_chunks):
        pos = full_text.find(chunk_text, search_cursor)
        if pos == -1:
            pos = full_text.find(chunk_text)
        
        if pos != -1:
            search_cursor = pos
            chunk_end = pos + len(chunk_text)
            # Find all pages this chunk overlaps with
            overlapping_pages = [
                p_num for (p_start, p_end, p_num) in page_spans
                if max(pos, p_start) < min(chunk_end, p_end)
            ]
            primary_page = overlapping_pages[0] if overlapping_pages else 1
            if len(overlapping_pages) > 1 and overlapping_pages[0] != overlapping_pages[-1]:
                page_label = f"{overlapping_pages[0]}-{overlapping_pages[-1]}"
            else:
                page_label = str(primary_page)
        else:
            primary_page = 1
            page_label = "1"

        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": clean_filename,
                "filename": clean_filename,
                "chunk_id": i,
                "page": primary_page,
                "page_label": page_label
            }
        )
        chunks.append(doc)

    # 7. Add to Chroma vector database
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
