import streamlit as st
from streamlit_app.utils.api_client import RAGAPIClient

st.set_page_config(
    page_title="Chat - Adaptive RAG",
    page_icon="💬",
    layout="wide",
)

# Initialize API client
api_client = RAGAPIClient()

# Initialize Streamlit chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session_1"

st.title("💬 Chat with Adaptive RAG")

# ─────────────────────────────────────────
# SIDEBAR: Document Upload Section
# ─────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document Management")
    st.markdown("Upload PDFs or TXT files to index into **Qdrant Vector DB**.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT"
    )

    doc_description = st.text_input(
        "Document Description (optional)",
        placeholder="e.g. Q3 Financial Report 2024"
    )

    if st.button("Upload & Index Document", type="primary", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Indexing document into Qdrant..."):
                try:
                    res = api_client.upload_document(
                        file_name=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                        description=doc_description
                    )
                    st.success(f"✅ Successfully indexed `{res['filename']}` into {res['chunks_indexed']} chunks!")
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
        else:
            st.warning("Please select a file first.")

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────
# MAIN CHAT INTERFACE
# ─────────────────────────────────────────

# Render past chat messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User chat input
if prompt := st.chat_input("Ask a question about your documents, web events, or general topics..."):
    # Display user message in UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call FastAPI backend and render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (Routing → Retrieving → Grading → Generating)"):
            try:
                response = api_client.query(
                    query_text=prompt,
                    session_id=st.session_state.session_id
                )
                answer = response["result"]["content"]
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"⚠️ Backend error: Could not connect to FastAPI server at http://localhost:8000. Is it running?\n\nDetails: {str(e)}"
                st.error(error_msg)