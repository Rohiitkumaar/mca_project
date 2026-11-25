import streamlit as st
from time import sleep

from core.config import ADMIN_USERNAME, ADMIN_PASSWORD
from core.vectorstore import (
    list_indexed_documents,
    get_total_chunks,
    index_pdf,
    delete_pdf,
    replace_pdf,
    reset_all_documents,
)


def render_admin_login():
    """Renders the admin login interface."""
    st.subheader("🔒 Admin Authentication")
    st.markdown("Please log in to manage the knowledge base.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("admin_login_form"):
            username = st.text_input("Username", value="")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Log In", use_container_width=True)

            if submit:
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state.admin_logged_in = True
                    st.success("Authentication successful! Redirecting...")
                    sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")


def render_admin_dashboard():
    """Renders the admin management dashboard for uploading, deleting, and replacing PDFs."""
    col_title, col_logout = st.columns([5, 1])
    with col_title:
        st.markdown("### 🛠️ Document Management Dashboard")
    with col_logout:
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.admin_logged_in = False
            st.session_state.current_view = "user"
            st.rerun()

    st.markdown("---")

    # Metrics Overview
    docs_list = list_indexed_documents()
    total_chunks = get_total_chunks()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Indexed PDFs", len(docs_list))
    m2.metric("Total Vector Chunks", total_chunks)
    m3.metric("Vector Database", "Persistent ChromaDB")

    st.markdown("---")

    # 1. Upload Section
    st.markdown("### 📤 Upload New PDF Documents")
    st.caption("Upload one or multiple PDFs to index and store into persistent memory.")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type="pdf",
        accept_multiple_files=True,
        key="admin_pdf_uploader"
    )

    if uploaded_files:
        if st.button(f"📥 Process & Index {len(uploaded_files)} File(s)", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {uploaded_file.name}...")
                chunks_indexed = index_pdf(uploaded_file.getvalue(), uploaded_file.name)
                st.write(f"✅ **{uploaded_file.name}**: Indexed {chunks_indexed} chunks.")
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("All documents processed successfully!")
            st.success("Documents successfully added to persistent ChromaDB!")
            sleep(1)
            st.rerun()

    st.markdown("---")

    # 2. Existing Documents Management
    st.markdown("### 📚 Active Knowledge Base Documents")
    if not docs_list:
        st.info("No documents are currently stored in the knowledge base. Upload PDFs above to begin.")
    else:
        for doc in docs_list:
            fname = doc["filename"]
            with st.expander(f"📄 **{fname}** — ({doc['chunks']} chunks | {doc['size_kb']} KB)", expanded=False):
                st.write(f"- **Filename:** `{fname}`")
                st.write(f"- **Indexed Chunks:** {doc['chunks']}")
                st.write(f"- **File Size:** {doc['size_kb']} KB")
                st.write(f"- **Last Modified:** {doc['modified_time']}")

                act_col1, act_col2 = st.columns(2)

                # Replace section
                with act_col1:
                    st.markdown("**🔄 Replace with Updated Version**")
                    replacement_file = st.file_uploader(
                        label=f"Choose updated PDF for {fname}",
                        type="pdf",
                        key=f"replace_{fname}"
                    )
                    if replacement_file:
                        if st.button(f"Update '{fname}'", key=f"btn_replace_{fname}"):
                            with st.spinner(f"Replacing {fname}..."):
                                new_chunks = replace_pdf(fname, replacement_file.getvalue())
                            st.success(f"Successfully replaced {fname} ({new_chunks} new chunks)!")
                            sleep(1)
                            st.rerun()

                # Delete section
                with act_col2:
                    st.markdown("**🗑️ Remove from Knowledge Base**")
                    st.caption("Deletes chunks from ChromaDB and removes the file.")
                    if st.button(f"Delete '{fname}'", type="secondary", key=f"btn_delete_{fname}"):
                        delete_pdf(fname)
                        st.success(f"Removed {fname} from persistent storage!")
                        sleep(1)
                        st.rerun()

    st.markdown("---")

    # 3. Master Reset
    with st.expander("⚠️ Danger Zone: Clear Entire Knowledge Base"):
        st.warning("This will permanently delete all indexed PDF documents, collections, and vector embeddings.")
        confirm = st.checkbox("I understand that this action cannot be undone", key="confirm_master_reset")
        if confirm:
            if st.button("🔥 Reset Entire Database", type="primary"):
                reset_all_documents()
                st.success("All documents and embeddings have been wiped cleanly.")
                sleep(1)
                st.rerun()


def render_admin_view():
    """Main entry for Admin View."""
    if not st.session_state.get("admin_logged_in", False):
        render_admin_login()
    else:
        render_admin_dashboard()
