import streamlit as st

from views.user_view import render_user_view
from views.admin_view import render_admin_view

# Streamlit Page Setup (No sidebar needed)
st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Session State Initialization
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_view" not in st.session_state:
    st.session_state.current_view = "user"


# User View Header with on-screen "Clear" and "Admin" buttons in the top right
if st.session_state.current_view == "user":
    col_title, col_clear, col_admin = st.columns([7, 1.2, 1.1])

    with col_title:
        st.subheader("💬 Your University Assistant")
        st.caption("Ask questions and get answers directly from the knowledge base.")

    with col_clear:
        st.write("")  # Spacing alignment
        if st.button("🧹 Clear", key="clear_chat_btn", help="Clear conversation history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_admin:
        st.write("")  # Spacing alignment
        if st.button("🔒 Admin", key="admin_nav_btn", help="Administrator Login", use_container_width=True):
            st.session_state.current_view = "admin"
            st.rerun()

    st.markdown("---")
    render_user_view()

# Admin View Header with "Chat" return button in the top right
else:
    col_title, col_chat = st.columns([8, 1.2])

    with col_title:
        st.subheader("🔒 Admin Portal")
        st.caption("Manage persistent documents and vector collections.")

    with col_chat:
        st.write("")  # Spacing alignment
        if st.button("💬 Chat", key="user_chat_nav_btn", help="Return to Chat UI", use_container_width=True):
            st.session_state.current_view = "user"
            st.rerun()

    st.markdown("---")
    render_admin_view()
