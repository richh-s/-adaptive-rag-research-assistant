import inspect
import logging
import time
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_assistant.graph.nodes.condense import condense_question
from rag_assistant.graph.nodes.corrective_fallback import corrective_web_search
from rag_assistant.graph.nodes.decompose import decompose_query, dispatch_retrieval
from rag_assistant.graph.nodes.fuse import fuse_results
from rag_assistant.graph.nodes.grade import after_grade, grade_and_score
from rag_assistant.graph.nodes.refine import refine_retrieval
from rag_assistant.graph.nodes.report import format_report
from rag_assistant.graph.nodes.retrieve import retrieve_bm25, retrieve_vector
from rag_assistant.graph.nodes.router import after_route, route_query
from rag_assistant.graph.nodes.synthesize import synthesize_answer
from rag_assistant.graph.nodes.verify import verify_groundedness
from rag_assistant.graph.nodes.web_search_node import web_search
from rag_assistant.graph.state import ResearchState
from rag_assistant.metrics import record_node_timing
from rag_assistant.tracing import span

logger = logging.getLogger(__name__)


def _finish(node_name: str, state: dict, result: dict, start: float) -> dict:
    """The bookkeeping both wrappers share, so the sync and async paths cannot drift in what
    they record."""
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    record_node_timing(node_name, elapsed_ms / 1000)
    logger.info(
        "node completed",
        extra={"trace_id": state.get("trace_id"), "node": node_name, "latency_ms": elapsed_ms},
    )
    return {**result, "node_timings": [{"node": node_name, "latency_ms": elapsed_ms}]}


def _span_kwargs(node_name: str, state: dict) -> dict:
    return {
        "rag.node": node_name,
        "rag.route": state.get("route"),
        "rag.trace_id": state.get("trace_id"),
    }


def _timed(node_name: str, node_fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Wraps a node function to record its own wall-clock latency into `node_timings` and log
    a structured line per invocation. Send-fanned nodes (retrieve_vector/retrieve_bm25/
    web_search) get invoked once per sub-query, so this contributes one entry per invocation,
    not one per node type -- the explainability panel sums/groups these by node name.

    Two wrappers, chosen by what it is wrapping. The graph is deliberately mixed: nodes whose
    work is an LLM call are `async def` and await it on the event loop, while nodes whose work
    is a blocking library call (Chroma, psycopg, the web-search client) stay synchronous and
    LangGraph runs them in a worker thread. Wrapping an async node in a sync wrapper would
    return the coroutine as the node's result, and the whole node would silently contribute
    nothing to the state -- so the shape of the wrapper has to follow the function.
    """
    if inspect.iscoroutinefunction(node_fn):

        async def async_wrapper(state: dict) -> dict:
            start = time.perf_counter()
            # One span per node invocation, nested under the request span from the API layer.
            # This is the only place that knows both the node's name and its boundaries, which
            # is why the instrumentation lives here rather than in each node -- twelve nodes
            # would otherwise each carry their own copy of it, and drift.
            with span(f"node.{node_name}", **_span_kwargs(node_name, state)):
                result = await node_fn(state)
            return _finish(node_name, state, result, start)

        return async_wrapper

    def wrapper(state: dict) -> dict:
        start = time.perf_counter()
        with span(f"node.{node_name}", **_span_kwargs(node_name, state)):
            result = node_fn(state)
        return _finish(node_name, state, result, start)

    return wrapper


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("condense_question", _timed("condense_question", condense_question))
    graph.add_node("route_query", _timed("route_query", route_query))
    graph.add_node("decompose_query", _timed("decompose_query", decompose_query))
    graph.add_node("retrieve_vector", _timed("retrieve_vector", retrieve_vector))
    graph.add_node("retrieve_bm25", _timed("retrieve_bm25", retrieve_bm25))
    graph.add_node("web_search", _timed("web_search", web_search))
    graph.add_node("fuse_results", _timed("fuse_results", fuse_results))
    graph.add_node("grade_and_score", _timed("grade_and_score", grade_and_score))
    graph.add_node("refine_retrieval", _timed("refine_retrieval", refine_retrieval))
    graph.add_node("corrective_web_search", _timed("corrective_web_search", corrective_web_search))
    graph.add_node("synthesize_answer", _timed("synthesize_answer", synthesize_answer))
    graph.add_node("verify_groundedness", _timed("verify_groundedness", verify_groundedness))
    graph.add_node("format_report", _timed("format_report", format_report))

    graph.add_edge(START, "condense_question")
    graph.add_edge("condense_question", "route_query")
    graph.add_conditional_edges(
        "route_query",
        after_route,
        ["decompose_query", "synthesize_answer"],
    )
    graph.add_conditional_edges(
        "decompose_query",
        dispatch_retrieval,
        ["retrieve_vector", "retrieve_bm25", "web_search"],
    )
    graph.add_edge("retrieve_vector", "fuse_results")
    graph.add_edge("retrieve_bm25", "fuse_results")
    graph.add_edge("web_search", "fuse_results")
    graph.add_edge("fuse_results", "grade_and_score")
    graph.add_conditional_edges(
        "grade_and_score",
        after_grade,
        ["refine_retrieval", "corrective_web_search", "synthesize_answer"],
    )
    # Both escalations rejoin at fusion rather than at synthesis, so a second attempt's
    # documents are ranked against the first attempt's instead of replacing them.
    graph.add_edge("refine_retrieval", "fuse_results")
    graph.add_edge("corrective_web_search", "fuse_results")
    # Verification sits between synthesis and the report, not inside synthesis: the answer
    # has already streamed to the client by this point, so the check delays the summary
    # rather than the prose, and a failure here cannot cost a request its answer.
    graph.add_edge("synthesize_answer", "verify_groundedness")
    graph.add_edge("verify_groundedness", "format_report")
    graph.add_edge("format_report", END)

    return graph.compile()
