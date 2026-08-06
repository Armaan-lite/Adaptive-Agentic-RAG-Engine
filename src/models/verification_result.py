from typing import Literal
from pydantic import BaseModel, Field


class GradeHallucination(BaseModel):
    """
    Binary grade schema to check if generation is grounded in document context.
    """
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Relevance score: 'yes' if generation is supported by/grounded in facts, 'no' if hallucinated."
    )


class GradeAnswer(BaseModel):
    """
    Binary grade schema to check if generation actually addresses the user's question.
    """
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Relevance score: 'yes' if generation addresses and answers the question, 'no' if not."
    )
