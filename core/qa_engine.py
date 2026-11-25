import os
import re
from typing import Dict, Any, List
from core.config import get_llm
from core.vectorstore import get_vector_store, get_total_chunks

def hybrid_search_and_rerank(vector_db, query: str, top_k: int = 8):
    """Performs candidate vector retrieval and applies BM25-style keyword, phrase, and document affinity boosting."""
    # Retrieve top 25 candidate chunks
    raw_results = vector_db.similarity_search_with_score(query, k=25)
    if not raw_results:
        return []

    query_lower = query.lower()
    stop_words = {
        'the', 'and', 'for', 'which', 'what', 'who', 'how', 'where', 'when',
        'is', 'are', 'was', 'were', 'in', 'of', 'on', 'to', 'with', 'a', 'an',
        'give', 'me', 'tell', 'about', 'explain', 'show', 'details', 'detail'
    }
    words = [
        w for w in re.findall(r'[a-zA-Z0-9_\.\-]+', query_lower)
        if len(w) > 1 and w not in stop_words
    ]

    reranked = []
    for doc, dist in raw_results:
        text = (doc.page_content or "").lower()
        filename = (doc.metadata.get('filename') or doc.metadata.get('source') or "").lower()

        # 1. Multi-word phrase matching bonus (e.g., "Web Technologies", ".NET Technology")
        phrase_bonus = 0.0
        for i in range(len(words) - 1):
            two_word_phrase = f"{words[i]} {words[i+1]}"
            if two_word_phrase in text:
                phrase_bonus += 0.30

        # 2. Individual keyword matching bonus
        kw_matches = sum(1 for w in words if w in text)

        # 3. Document affinity bonus (e.g., user asks for 'mca' and document is 'mca syllabbus.pdf')
        doc_affinity_bonus = 0.0
        for w in words:
            if w in filename:
                doc_affinity_bonus += 0.25

        # Hybrid composite score (lower is more relevant)
        hybrid_score = dist - (0.07 * kw_matches) - phrase_bonus - doc_affinity_bonus
        reranked.append((doc, hybrid_score, dist))

    # Sort strictly by hybrid relevance score
    reranked.sort(key=lambda x: x[1])
    return reranked[:top_k]


from langchain_core.documents import Document

def query_knowledge_base(query: str, k: int = 5) -> Dict[str, Any]:
    """Retrieves relevant chunks from ChromaDB using Hybrid Re-ranking, expands contiguous multi-page neighbor chunks, and generates a complete answer with Gemini."""
    total_chunks = get_total_chunks()
    if total_chunks == 0:
        return {
            "answer": "No documents are currently indexed in the knowledge base. Please contact an administrator to upload PDF documents.",
            "sources": [],
            "has_context": False
        }

    vector_db = get_vector_store()
    retrieved_docs_with_scores = hybrid_search_and_rerank(vector_db, query, top_k=k)

    if not retrieved_docs_with_scores:
        return {
            "answer": "I could not find any relevant information in the uploaded documents to answer your question.",
            "sources": [],
            "has_context": False
        }

    # 1. Contiguous Window Expansion:
    # Capture complete continuous syllabus multi-page spans (Unit I, II, III, IV, Books)
    expanded_target_keys = set()

    for i, (doc, hybrid_score, orig_dist) in enumerate(retrieved_docs_with_scores):
        src = doc.metadata.get("source") or doc.metadata.get("filename") or ""
        cid = doc.metadata.get("chunk_id", 0)
        
        # Expand top matches with adjacent contiguous chunks (-1 before, up to +4 after)
        # to ensure all subsequent units, topics, and reference books across pages are included
        neighbor_offsets = list(range(-1, 5)) if i < 3 else [0, 1, 2]
        for offset in neighbor_offsets:
            if cid + offset >= 0:
                expanded_target_keys.add((src, cid + offset))

    # Fetch all chunks from the Chroma collection to build seamless continuous text
    raw_col_data = vector_db._collection.get(include=["documents", "metadatas"])
    expanded_docs: List[Document] = []
    
    for text, meta in zip(raw_col_data["documents"], raw_col_data["metadatas"]):
        chunk_key = (meta.get("source") or meta.get("filename"), meta.get("chunk_id"))
        if chunk_key in expanded_target_keys:
            expanded_docs.append(Document(page_content=text, metadata=meta))

    # 2. Extract ranked sources (reflecting the exact pages covered by the expanded continuous context)
    sources: List[Dict[str, Any]] = []
    seen_sources = set()
    rank = 1

    # First add the primary matches in order of hybrid relevance rank
    for doc, hybrid_score, orig_dist in retrieved_docs_with_scores:
        clean_src = os.path.basename(doc.metadata.get("source") or doc.metadata.get("filename") or "Unknown Document")
        cid = doc.metadata.get("chunk_id", 0)
        
        # Check all related pages covered by the continuous chunk span
        related_pages = []
        for offset in range(0, 5):
            for edoc in expanded_docs:
                if (edoc.metadata.get("source") == doc.metadata.get("source") and 
                    edoc.metadata.get("chunk_id") == cid + offset):
                    p = str(edoc.metadata.get("page_label") or edoc.metadata.get("page", 1))
                    if p not in related_pages:
                        related_pages.append(p)
        
        if len(related_pages) > 1 and related_pages[0] != related_pages[-1]:
            page_label = f"{related_pages[0]}-{related_pages[-1]}"
        elif related_pages:
            page_label = related_pages[0]
        else:
            page_label = str(doc.metadata.get("page_label") or doc.metadata.get("page", 1))

        source_key = f"{clean_src}_p{page_label}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": clean_src,
                "page": page_label,
                "rank": rank,
                "is_best_match": (rank == 1),
                "score": round(float(hybrid_score), 4)
            })
            rank += 1

    # 3. Sort expanded chunks in natural chronological reading order for LLM synthesis
    expanded_docs.sort(
        key=lambda d: (
            d.metadata.get("filename") or d.metadata.get("source") or "",
            d.metadata.get("chunk_id", 0)
        )
    )

    context_parts = []
    for doc in expanded_docs:
        clean_src = os.path.basename(doc.metadata.get("source") or doc.metadata.get("filename") or "Unknown Document")
        page_label = doc.metadata.get("page_label") or str(doc.metadata.get("page", 1))

        context_parts.append(
            f"--- Document: {clean_src} (Page {page_label}, Chunk #{doc.metadata.get('chunk_id', 0)}) ---\n{doc.page_content.strip()}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful and highly accurate document assistant. 
Answer the user's question clearly, comprehensively, and thoroughly based strictly on the provided document context.

Important Guidelines:
1. The context excerpts are chronologically ordered continuous sections of the document.
2. If a syllabus, topic, or section starts on one page and continues across subsequent pages (e.g. Unit I on first page, Unit II, III, IV on subsequent pages), include ALL units, topics, and details in complete order.
3. Do not omit any units or continuous details that appear in the context.

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
