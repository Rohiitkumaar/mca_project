import os
from typing import Dict, Any, List
from core.config import get_llm
from core.vectorstore import get_vector_store, get_total_chunks


def query_knowledge_base(query: str, k: int = 5) -> Dict[str, Any]:
    """Retrieves relevant chunks from ChromaDB and generates an answer with source citations using Gemini."""
    total_chunks = get_total_chunks()
    if total_chunks == 0:
        return {
            "answer": "No documents are currently indexed in the knowledge base. Please contact an administrator to upload PDF documents.",
            "sources": [],
            "has_context": False
        }

    vector_db = get_vector_store()
    retrieved_docs = vector_db.similarity_search(query, k=k)

    if not retrieved_docs:
        return {
            "answer": "I could not find any relevant information in the uploaded documents to answer your question.",
            "sources": [],
            "has_context": False
        }

    context_parts = []
    sources: List[Dict[str, Any]] = []
    seen_sources = set()

    for doc in retrieved_docs:
        raw_src = doc.metadata.get("source") or doc.metadata.get("filename") or "Unknown Document"
        clean_src = os.path.basename(raw_src)
        page = doc.metadata.get("page", 0)
        # 1-indexed page for human display
        display_page = page + 1 if isinstance(page, int) else page

        source_key = f"{clean_src}_p{display_page}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": clean_src,
                "page": display_page
            })

        context_parts.append(f"--- Document: {clean_src} (Page {display_page}) ---\n{doc.page_content.strip()}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful and knowledgeable document assistant. 
Answer the user's question clearly and accurately based strictly on the provided document context.
If the context does not contain enough information to answer the question, state that politely.

Context:
{context}

Question:
{query}

Answer:"""

    llm = get_llm()
    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        answer = "\n".join(
            block["text"]
            for block in response.content
            if block.get("type") == "text"
        )
    else:
        answer = str(response.content)

    return {
        "answer": answer,
        "sources": sources,
        "has_context": True
    }
