import streamlit as st
from core.qa_engine import query_knowledge_base


def render_source_item(src: dict) -> str:
    """Helper to format source citation with rank indicator."""
    rank = src.get("rank", 1)
    filename = src.get("source", "Document")
    page = src.get("page", 1)
    
    if src.get("is_best_match") or rank == 1:
        return f"🏆 **Best Match (Rank #1)**: `{filename}` (Page {page})"
    elif rank == 2:
        return f"🥈 **Rank #2**: `{filename}` (Page {page})"
    elif rank == 3:
        return f"🥉 **Rank #3**: `{filename}` (Page {page})"
    else:
        return f"📄 **Rank #{rank}**: `{filename}` (Page {page})"


def render_user_view():
    """Renders the User Chat interface showing ranked PDF sources with best matches first."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📌 Sources (Ranked by Best Match)", expanded=False):
                    for src in message["sources"]:
                        st.markdown(render_source_item(src))

    query = st.chat_input("Ask a question...")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking & generating answer..."):
            result = query_knowledge_base(query)
            answer = result["answer"]
            sources = result.get("sources", [])

        with st.chat_message("ai"):
            st.markdown(answer)
            if sources:
                with st.expander("📌 Sources (Ranked by Best Match)", expanded=False):
                    for src in sources:
                        st.markdown(render_source_item(src))

        st.session_state.messages.append({
            "role": "ai",
            "content": answer,
            "sources": sources
        })
