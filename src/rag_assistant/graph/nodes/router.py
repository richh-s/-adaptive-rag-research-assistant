import re
from pathlib import Path

from rag_assistant.cache import cache_get, cache_key, cache_set
from rag_assistant.config import get_settings
from rag_assistant.graph.state import ResearchState
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.llm import get_structured_llm
from rag_assistant.prompts.router_prompt import ROUTER_PROMPT
from rag_assistant.schemas.models import RouteDecision

_HASH_SUFFIX_RE = re.compile(r"_[0-9a-f]{8}$")


def _describe_local_corpus() -> str:
    """Turns the ingestion manifest's indexed filenames into a human-readable topic list, so
    the router judges routes against what's actually indexed instead of a hardcoded, stale
    description -- otherwise a newly uploaded document outside the original topic set always
    gets misrouted to web search even though it was correctly indexed locally."""
    manifest = load_manifest(get_settings().chroma_persist_dir)
    if not manifest:
        return "(empty -- no documents indexed yet)"

    topics = []
    for filename in sorted(manifest):
        stem = _HASH_SUFFIX_RE.sub("", Path(filename).stem)
        topics.append(stem.replace("_", " ").replace("-", " ").strip())
    return "; ".join(topics)


def route_query(state: ResearchState) -> dict:
    """Agentic/Self-RAG routing: ask the LLM which retrieval path(s) this question needs,
    instead of always retrieving the same way. Routing depends on both the question text and
    the current corpus contents, so the cache key covers both -- a freshly uploaded document
    must not be masked by a decision cached before it existed."""
    question = state["question"]
    corpus_description = _describe_local_corpus()
    key = cache_key("router", question, corpus_description)
    cached = cache_get(key)
    if cached is not None:
        return {"route": cached["route"], "route_reasoning": cached["route_reasoning"]}

    llm = get_structured_llm(RouteDecision)
    decision: RouteDecision = llm.invoke(
        ROUTER_PROMPT.format(question=question, corpus_description=corpus_description)
    )
    result = {"route": decision.route, "route_reasoning": decision.reasoning}
    cache_set(key, result, get_settings().cache_ttl_router)
    return result


def after_route(state: ResearchState) -> str:
    """Conditional edge function: skip decomposition entirely when no retrieval is
    needed -- there's nothing to decompose a query for if we won't retrieve anything."""
    return "synthesize_answer" if state["route"] == "none" else "decompose_query"
