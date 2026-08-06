import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file into os.environ
load_dotenv()


class Settings(BaseModel):
    """
    Application Settings and Environment Configuration Manager.
    """
    # OpenAI Credentials & Models
    OPENAI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="API Key for OpenAI services"
    )
    MODEL_NAME: str = Field(
        default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"),
        description="Default LLM model name for chat and routing"
    )
    EMBEDDING_MODEL: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        description="Default OpenAI embedding model for vector search"
    )

    # Qdrant Vector Database
    QDRANT_URL: Optional[str] = Field(
        default_factory=lambda: os.getenv("QDRANT_URL", None),
        description="URL for Qdrant server. If empty, runs in local embedded mode"
    )
    QDRANT_API_KEY: Optional[str] = Field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY", None),
        description="API Key for Qdrant Cloud (optional)"
    )

    # Tavily Web Search
    TAVILY_API_KEY: Optional[str] = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", None),
        description="API Key for Tavily Web Search API"
    )

    # MongoDB Session Storage
    MONGODB_URI: str = Field(
        default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        description="MongoDB connection string"
    )
    MONGODB_DB_NAME: str = Field(
        default_factory=lambda: os.getenv("MONGODB_DB_NAME", "adaptive_rag"),
        description="MongoDB database name"
    )


# Instantiate a global singleton settings object
settings = Settings()