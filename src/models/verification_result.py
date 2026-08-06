from pydantic import BaseModel, Field


class AuditResult(BaseModel):
    """
    Combined Self-RAG audit schema evaluating hallucination and question relevance in a single pass.
    """
    is_grounded: bool = Field(
        ...,
        description="True if generation is strictly grounded in and supported by provided facts, False if hallucinated."
    )
    answers_question: bool = Field(
        ...,
        description="True if generation addresses and answers the user's question, False if not."
    )


class GradeDocuments(BaseModel):
    """
    Binary grade schema for evaluating document relevance.
    """
    binary_score: str = Field(
        ...,
        description="Relevance score: 'yes' if document is relevant to query, 'no' if not."
    )
