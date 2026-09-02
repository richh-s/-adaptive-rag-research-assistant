from langgraph.types import Send

from rag_assistant.auth import PUBLIC_OWNER

from rag_assistant.graph.state import ResearchState
from rag_assistant.llm import get_structured_llm
from rag_assistant.prompts.decompose_prompt import DECOMPOSE_PROMPT
from rag_assistant.schemas.models import SubQueries


def decompose_query(state: ResearchState) -> dict:
    """Query decomposition: split a compound question into focused sub-queries so each
    retrieval pass targets one thing instead of one averaged embedding for everything at
    once. Simple questions pass through as a single-element list, so every downstream node
    can assume a uniform "list of sub-queries" shape regardless of question complexity."""
    llm = get_structured_llm(SubQueries)
    result: SubQueries = llm.invoke(DECOMPOSE_PROMPT.format(question=state["question"]))
    return {"sub_queries": result.sub_queries}


def dispatch_retrieval(state: ResearchState) -> list[Send]:
    """Fans out one `Send` per (sub-query, retrieval path) pair -- LangGraph's map step.
    Each `Send` triggers an independent invocation of `retrieve_vector`/`retrieve_bm25`/
    `web_search` carrying only that one sub-query; their `vector_results`/`bm25_results`/
    `web_results` writes are concatenated back together via the `operator.add` reducer
    declared on those state fields."""
    route = state["route"]
    # A `Send` payload replaces the state for that invocation rather than extending it, so
    # the owner has to be copied in explicitly -- a retrieval node that read `state["owner"]`
    # without this would raise KeyError, and one that defaulted to "public" would silently
    # serve every tenant the shared corpus only.
    owner = state.get("owner") or PUBLIC_OWNER
    sends = []
    for sub_query in state["sub_queries"]:
        payload = {"sub_query": sub_query, "owner": owner}
        if route in ("vector", "both"):
            sends.append(Send("retrieve_vector", payload))
            sends.append(Send("retrieve_bm25", payload))
        if route in ("web", "both"):
            sends.append(Send("web_search", payload))
    return sends
