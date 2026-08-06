import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.config.settings import settings
from src.llms.openai import get_embeddings

COLLECTION_NAME = "adaptive_rag_documents"


def get_qdrant_client() -> QdrantClient:
    """
    Returns an initialized QdrantClient.
    If QDRANT_URL is configured, connects to server/cloud.
    Otherwise, runs in local embedded in-memory/file mode (100% Free!).
    """
    if settings.QDRANT_URL and settings.QDRANT_URL.strip():
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    else:
        # Embedded local storage mode inside .qdrant directory
        local_path = os.path.join(os.getcwd(), ".qdrant")
        return QdrantClient(path=local_path)


def get_vector_store() -> QdrantVectorStore:
    """
    Initializes and returns the QdrantVectorStore with OpenAI embeddings.
    """
    client = get_qdrant_client()
    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def get_retriever(k: int = 4):
    """
    Returns a LangChain Retriever configured to fetch the top k most relevant document chunks.

    Args:
        k: Number of relevant chunks to retrieve (default: 4).
    """
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})