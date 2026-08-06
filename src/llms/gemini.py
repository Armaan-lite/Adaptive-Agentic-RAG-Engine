from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.config.settings import settings


def get_llm(temperature: float = 0.0, model_name: str = None) -> ChatGoogleGenerativeAI:
    """
    Factory function to return an initialized ChatGoogleGenerativeAI (Gemini) instance.

    Args:
        temperature: Controls randomness in generation (0.0 = deterministic).
        model_name: Optional override for the Gemini model name.

    Returns:
        ChatGoogleGenerativeAI: Configured LangChain Gemini chat client.
    """
    model = model_name or settings.MODEL_NAME
    if model == "gpt-4o-mini" or not model:
        model = "gemini-flash-latest"

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=settings.GEMINI_API_KEY,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Factory function to return an initialized GoogleGenerativeAIEmbeddings instance.

    Returns:
        GoogleGenerativeAIEmbeddings: Configured LangChain Gemini embeddings client.
    """
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
    )
