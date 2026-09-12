import streamlit as st
from core.qa_engine import query_knowledge_base


def render_user_view():
    """Renders the User Chat interface showing only clean PDF filename and page number for sources."""
    # Initialize conversation state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📌 Sources", expanded=False):
                    for src in message["sources"]:
                        st.markdown(f"📄 `{src['source']}` (Page {src['page']})")

    # Chat input
    query = st.chat_input("Ask a question...")
    if query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Generate answer via QA engine
        with st.spinner("Thinking & generating answer..."):
            result = query_knowledge_base(query)
            answer = result["answer"]
            sources = result.get("sources", [])

        # Display AI message with only filename and page number
        with st.chat_message("ai"):
            st.markdown(answer)
            if sources:
                with st.expander("📌 Sources", expanded=False):
                    for src in sources:
                        st.markdown(f"📄 `{src['source']}` (Page {src['page']})")

        # Save to history
        st.session_state.messages.append({
            "role": "ai",
            "content": answer,
            "sources": sources
        })
