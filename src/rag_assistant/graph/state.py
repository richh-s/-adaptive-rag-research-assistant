import operator
from typing import Annotated, Literal, TypedDict

from rag_assistant.schemas.models import SubQueryResult


class ResearchState(TypedDict):
    """Shared state threaded through every node in the graph. Each node reads what it
    needs and returns a partial dict of updates; LangGraph merges that into this state."""

    question: str

    # routing -- Concept: Agentic/Self-RAG
    route: Literal["vector", "web", "both", "none"] | None
    route_reasoning: str | None

    # decomposition -- Concept: query decomposition
    sub_queries: list[str]

    # `operator.add` reducer: each Send-based retrieve_vector/web_search invocation
    # contributes a one-element list for its sub-query, and LangGraph concatenates them
    # all here instead of the default "last write wins" behavior.
    vector_results: Annotated[list[SubQueryResult], operator.add]
    web_results: Annotated[list[SubQueryResult], operator.add]

    final_answer: str
    citations: list[dict]

    errors: list[str]
