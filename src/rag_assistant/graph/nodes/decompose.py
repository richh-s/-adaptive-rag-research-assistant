from langgraph.types import Send

from rag_assistant.graph.state import ResearchState
from rag_assistant.llm import get_chat_model
from rag_assistant.prompts.decompose_prompt import DECOMPOSE_PROMPT
from rag_assistant.schemas.models import SubQueries


def decompose_query(state: ResearchState) -> dict:
    """Query decomposition: split a compound question into focused sub-queries so each
    retrieval pass targets one thing instead of one averaged embedding for everything at
    once. Simple questions pass through as a single-element list, so every downstream node
    can assume a uniform "list of sub-queries" shape regardless of question complexity."""
    llm = get_chat_model().with_structured_output(SubQueries)
    result: SubQueries = llm.invoke(DECOMPOSE_PROMPT.format(question=state["question"]))
    return {"sub_queries": result.sub_queries}


def dispatch_retrieval(state: ResearchState) -> list[Send]:
    """Fans out one `Send` per (sub-query, retrieval path) pair -- LangGraph's map step.
    Each `Send` triggers an independent invocation of `retrieve_vector`/`web_search` carrying
    only that one sub-query; their `vector_results`/`web_results` writes are concatenated back
    together via the `operator.add` reducer declared on those state fields."""
    route = state["route"]
    sends = []
    for sub_query in state["sub_queries"]:
        if route in ("vector", "both"):
            sends.append(Send("retrieve_vector", {"sub_query": sub_query}))
        if route in ("web", "both"):
            sends.append(Send("web_search", {"sub_query": sub_query}))
    return sends
