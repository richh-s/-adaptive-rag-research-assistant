import time
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_assistant.graph.nodes.corrective_fallback import corrective_web_search
from rag_assistant.graph.nodes.decompose import decompose_query, dispatch_retrieval
from rag_assistant.graph.nodes.fuse import fuse_results
from rag_assistant.graph.nodes.grade import after_grade, grade_and_score
from rag_assistant.graph.nodes.report import format_report
from rag_assistant.graph.nodes.retrieve import retrieve_bm25, retrieve_vector
from rag_assistant.graph.nodes.router import after_route, route_query
from rag_assistant.graph.nodes.synthesize import synthesize_answer
from rag_assistant.graph.nodes.web_search_node import web_search
from rag_assistant.graph.state import ResearchState


def _timed(node_name: str, node_fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Wraps a node function to record its own wall-clock latency into `node_timings`.
    Send-fanned nodes (retrieve_vector/retrieve_bm25/web_search) get invoked once per
    sub-query, so this contributes one entry per invocation, not one per node type --
    the explainability panel sums/groups these by node name when it builds the summary."""

    def wrapper(state: dict) -> dict:
        start = time.perf_counter()
        result = node_fn(state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {**result, "node_timings": [{"node": node_name, "latency_ms": round(elapsed_ms, 1)}]}

    return wrapper


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("route_query", _timed("route_query", route_query))
    graph.add_node("decompose_query", _timed("decompose_query", decompose_query))
    graph.add_node("retrieve_vector", _timed("retrieve_vector", retrieve_vector))
    graph.add_node("retrieve_bm25", _timed("retrieve_bm25", retrieve_bm25))
    graph.add_node("web_search", _timed("web_search", web_search))
    graph.add_node("fuse_results", _timed("fuse_results", fuse_results))
    graph.add_node("grade_and_score", _timed("grade_and_score", grade_and_score))
    graph.add_node("corrective_web_search", _timed("corrective_web_search", corrective_web_search))
    graph.add_node("synthesize_answer", _timed("synthesize_answer", synthesize_answer))
    graph.add_node("format_report", _timed("format_report", format_report))

    graph.add_edge(START, "route_query")
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
        ["corrective_web_search", "synthesize_answer"],
    )
    graph.add_edge("corrective_web_search", "fuse_results")
    graph.add_edge("synthesize_answer", "format_report")
    graph.add_edge("format_report", END)

    return graph.compile()
