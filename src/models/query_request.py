from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Schema for incoming REST API query requests.
    """
    query: str = Field(..., description="User question or prompt", min_length=1)
    session_id: Optional[str] = Field(default="default_session", description="Session ID for chat memory")