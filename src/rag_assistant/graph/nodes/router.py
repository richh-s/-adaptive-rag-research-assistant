from rag_assistant.cache import cache_get, cache_key, cache_set
from rag_assistant.config import get_settings
from rag_assistant.graph.state import ResearchState
from rag_assistant.llm import get_structured_llm
from rag_assistant.prompts.router_prompt import ROUTER_PROMPT
from rag_assistant.schemas.models import RouteDecision


def route_query(state: ResearchState) -> dict:
    """Agentic/Self-RAG routing: ask the LLM which retrieval path(s) this question needs,
    instead of always retrieving the same way. Routing is a pure function of the question
    text, so identical questions can safely share a cached decision for a few minutes."""
    question = state["question"]
    key = cache_key("router", question)
    cached = cache_get(key)
    if cached is not None:
        return {"route": cached["route"], "route_reasoning": cached["route_reasoning"]}

    llm = get_structured_llm(RouteDecision)
    decision: RouteDecision = llm.invoke(ROUTER_PROMPT.format(question=question))
    result = {"route": decision.route, "route_reasoning": decision.reasoning}
    cache_set(key, result, get_settings().cache_ttl_router)
    return result


def after_route(state: ResearchState) -> str:
    """Conditional edge function: skip decomposition entirely when no retrieval is
    needed -- there's nothing to decompose a query for if we won't retrieve anything."""
    return "synthesize_answer" if state["route"] == "none" else "decompose_query"
