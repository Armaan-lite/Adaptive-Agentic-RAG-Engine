from typing import Literal
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """
    Binary grade schema for evaluating document relevance.
    """
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Relevance score: 'yes' if document is relevant to query, 'no' if not."
    )