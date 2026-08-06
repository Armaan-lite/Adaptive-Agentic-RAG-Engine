import streamlit as st

st.set_page_config(
    page_title="Adaptive Agentic RAG Engine",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Adaptive Agentic RAG Engine")
st.subheader("Production-Grade Self-Corrective RAG powered by LangGraph, Qdrant & FastAPI")

st.markdown("""
---
### 🚀 System Architecture Overview

This application features an **Agentic Adaptive RAG** engine that dynamically routes and self-corrects queries using a LangGraph state machine:

1. **🧠 Query Router Agent**: Automatically classifies user queries into **Vector DB** (custom documents), **Web Search** (real-time news/events), or **General Knowledge**.
2. **📚 Vector Search (Qdrant)**: Embeds and retrieves semantically relevant document chunks using `text-embedding-3-small`.
3. **🔍 Document Grader Agent**: Evaluates retrieved chunks for relevance. If chunks are irrelevant, it triggers a query rewrite.
4. **✏️ Query Rewriter Agent**: Reformulates vague queries into search-engine optimized prompts.
5. **🌐 Web Search Fallback (Tavily)**: Automatically searches the live web when custom documents do not contain the answer.

---
### 🛠️ Technology Stack
- **Orchestration**: LangGraph / LangChain
- **LLM & Embeddings**: OpenAI `gpt-4o-mini` & `text-embedding-3-small`
- **Vector DB**: Qdrant (Embedded Local Storage)
- **Backend API**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Web Search**: Tavily Search API
---
👈 **Use the sidebar on the left to navigate to the Chat Interface!**
""")