from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config.settings import settings


def get_llm(temperature: float = 0.0, model_name: str = None) -> ChatOpenAI:
    """
    Factory function to return an initialized ChatOpenAI instance.

    Args:
        temperature: Controls randomness in generation (0.0 = deterministic).
        model_name: Optional override for the LLM model name.

    Returns:
        ChatOpenAI: Configured LangChain OpenAI chat client.
    """
    model = model_name or settings.MODEL_NAME
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """
    Factory function to return an initialized OpenAIEmbeddings instance.

    Returns:
        OpenAIEmbeddings: Configured LangChain OpenAI embeddings client.
    """
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )