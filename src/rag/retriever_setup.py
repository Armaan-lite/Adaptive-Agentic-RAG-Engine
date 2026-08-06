import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.config.settings import settings
from src.llms.groq_client import get_embeddings

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
    Initializes and returns the QdrantVectorStore with local HuggingFace embeddings.
    Auto-detects vector dimension mismatches and recreates collection cleanly.
    """
    client = get_qdrant_client()
    embeddings = get_embeddings()

    # Determine embedding vector dimension (e.g. 384 for all-MiniLM-L6-v2)
    sample_vector = embeddings.embed_query("test")
    vector_size = len(sample_vector)

    from qdrant_client.http import models as rest

    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        existing_size = info.config.params.vectors.size
        if existing_size != vector_size:
            print(f"--- DIMENSION MISMATCH DETECTED (Existing: {existing_size}, New: {vector_size}) -> Resetting Collection ---")
            client.delete_collection(COLLECTION_NAME)
            client = get_qdrant_client()
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE,
                ),
            )
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )

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