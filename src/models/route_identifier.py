from typing import Literal
from pydantic import BaseModel, Field

class RouteQuery(BaseModel):
    """
    Pydantic schema to strictly route a user query to the correct datasource.
    """
    datasource: Literal["vectorstore","general","websearch"]=Field(
        ...,
        description=(
            "Select the best datasource for the query: "
            "'vectorstore' for queries about uploaded custom documents, "
            "'websearch' for real-time topics or current events, "
            "'general' for general knowledge, greetings, or basic conversation."
        ),
    )