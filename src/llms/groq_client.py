from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.config.settings import settings


def get_llm(temperature: float = 0.0, model_name: str = None) -> ChatGroq:
    """
    Factory function to return an initialized ChatGroq (Llama-3.3-70B) instance.

    Args:
        temperature: Controls randomness in generation (0.0 = deterministic).
        model_name: Optional override for the Groq model name.

    Returns:
        ChatGroq: Configured LangChain Groq chat client.
    """
    model = model_name or settings.MODEL_NAME
    
    # Automatically map legacy/other model names to Groq Llama-3.3-70B
    if model in ["gpt-4o-mini", "gemini-2.0-flash", "gemini-flash-latest", "gemini-1.5-flash"] or not model:
        model = "llama-3.3-70b-versatile"

    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=settings.GROQ_API_KEY,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Factory function to return an initialized 100% local HuggingFaceEmbeddings instance.

    Returns:
        HuggingFaceEmbeddings: Configured local embeddings model.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL
    )
