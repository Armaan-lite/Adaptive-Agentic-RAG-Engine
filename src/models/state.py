from typing import List, Optional
from typing_extensions import TypedDict
class GraphState(TypedDict):
    """
    Represents the shared state passed between nodes in our LangGraph workflow.
    """
    query: str
    generation: str
    web_search: bool
    documents: List[str]
    session_id: Optional[str]
    retry_count: Optional[int]