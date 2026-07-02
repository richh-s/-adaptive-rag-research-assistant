from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_assistant.graph.nodes.corrective_fallback import corrective_web_search
from rag_assistant.graph.nodes.decompose import decompose_query, dispatch_retrieval
from rag_assistant.graph.nodes.fuse import fuse_results
from rag_assistant.graph.nodes.grade import after_grade, grade_and_score
from rag_assistant.graph.nodes.retrieve import retrieve_vector
from rag_assistant.graph.nodes.router import after_route, route_query
from rag_assistant.graph.nodes.synthesize import synthesize_answer
from rag_assistant.graph.nodes.web_search_node import web_search
from rag_assistant.graph.state import ResearchState


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("route_query", route_query)
    graph.add_node("decompose_query", decompose_query)
    graph.add_node("retrieve_vector", retrieve_vector)
    graph.add_node("web_search", web_search)
    graph.add_node("fuse_results", fuse_results)
    graph.add_node("grade_and_score", grade_and_score)
    graph.add_node("corrective_web_search", corrective_web_search)
    graph.add_node("synthesize_answer", synthesize_answer)

    graph.add_edge(START, "route_query")
    graph.add_conditional_edges(
        "route_query",
        after_route,
        ["decompose_query", "synthesize_answer"],
    )
    graph.add_conditional_edges(
        "decompose_query",
        dispatch_retrieval,
        ["retrieve_vector", "web_search"],
    )
    graph.add_edge("retrieve_vector", "fuse_results")
    graph.add_edge("web_search", "fuse_results")
    graph.add_edge("fuse_results", "grade_and_score")
    graph.add_conditional_edges(
        "grade_and_score",
        after_grade,
        ["corrective_web_search", "synthesize_answer"],
    )
    graph.add_edge("corrective_web_search", "fuse_results")
    graph.add_edge("synthesize_answer", END)

    return graph.compile()
