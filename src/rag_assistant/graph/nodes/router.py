from rag_assistant.graph.state import ResearchState
from rag_assistant.llm import get_chat_model
from rag_assistant.prompts.router_prompt import ROUTER_PROMPT
from rag_assistant.schemas.models import RouteDecision


def route_query(state: ResearchState) -> dict:
    """Agentic/Self-RAG routing: ask the LLM which retrieval path(s) this question needs,
    instead of always retrieving the same way."""
    llm = get_chat_model().with_structured_output(RouteDecision)
    decision: RouteDecision = llm.invoke(ROUTER_PROMPT.format(question=state["question"]))
    return {"route": decision.route, "route_reasoning": decision.reasoning}


def after_route(state: ResearchState) -> str:
    """Conditional edge function: skip decomposition entirely when no retrieval is
    needed -- there's nothing to decompose a query for if we won't retrieve anything."""
    return "synthesize_answer" if state["route"] == "none" else "decompose_query"
