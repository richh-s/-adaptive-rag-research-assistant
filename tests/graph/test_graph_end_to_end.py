"""The compiled graph, run offline, with every external call stubbed.

The wiring this covers has no other test. The graph is deliberately mixed: nodes whose work
is an LLM call are coroutines and await it on the event loop, while nodes whose work is a
blocking library call stay synchronous and LangGraph runs them in a worker thread. Getting
that wrong does not raise -- a coroutine returned from a sync wrapper is a perfectly good
object that simply contributes nothing to the state -- so the node would silently do nothing
and the answer would be built from whatever the previous node left behind.

Everything else here is a unit test of one node with its neighbours mocked out. This is the
only place the real `build_graph()` topology executes end to end, which is also what makes it
the only test that would catch an edge wired to the wrong node.
"""

import pytest

from rag_assistant.graph import build_graph as build_graph_module
from rag_assistant.schemas.models import (
    CondensedQuestion,
    DocGrade,
    GroundednessReport,
    RetrievedDoc,
    RouteDecision,
    SubQueries,
)


class _StubLLM:
    """Returns whatever the schema it was asked for needs, so one stub serves every node."""

    def __init__(self, schema=None):
        self.schema = schema

    async def ainvoke(self, prompt, **kwargs):
        if self.schema is RouteDecision:
            return RouteDecision(route="vector", reasoning="the corpus covers this")
        if self.schema is SubQueries:
            return SubQueries(sub_queries=["who founded it", "what does it build"])
        if self.schema is CondensedQuestion:
            return CondensedQuestion(standalone_question="Who founded Anthropic?")
        if self.schema is GroundednessReport:
            return GroundednessReport(claims=[])
        if self.schema is not None:  # DocGradeBatch
            return self.schema(grades=[DocGrade(relevant=True, score=0.9)] * 2)
        return type(
            "Answer", (), {"text": "Anthropic was founded by the Amodeis [1].", "content": ""}
        )()


@pytest.fixture
def offline_graph(monkeypatch):
    """Stubs every boundary the graph touches: both LLM factories and both retrieval paths."""
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.router.get_structured_llm", lambda s, **k: _StubLLM(s)
    )
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.decompose.get_structured_llm", lambda s, **k: _StubLLM(s)
    )
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.condense.get_structured_llm", lambda s, **k: _StubLLM(s)
    )
    monkeypatch.setattr(
        "rag_assistant.grading.relevance_grader.get_structured_llm", lambda s, **k: _StubLLM(s)
    )
    monkeypatch.setattr(
        "rag_assistant.grading.groundedness.get_structured_llm", lambda s, **k: _StubLLM(s)
    )
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.synthesize.get_chat_model", lambda **k: _StubLLM()
    )
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.router._describe_local_corpus",
        lambda owner="public", principals=None: "anthropic",
    )

    class _Retriever:
        def invoke(self, query):
            from langchain_core.documents import Document

            return [
                Document(
                    page_content="Anthropic was founded by Dario and Daniela Amodei.",
                    metadata={"source": "anthropic.md", "owner": "public"},
                )
            ]

    monkeypatch.setattr(
        "rag_assistant.graph.nodes.retrieve.get_retriever", lambda **k: _Retriever()
    )
    monkeypatch.setattr(
        "rag_assistant.graph.nodes.retrieve.bm25_search",
        lambda q, **k: [
            RetrievedDoc(
                content="Anthropic builds the Claude family.",
                metadata={"source": "anthropic.md"},
                source_id="anthropic.md",
            )
        ],
    )
    return build_graph_module.build_graph()


async def test_the_whole_graph_runs_and_produces_a_cited_report(offline_graph):
    result = await offline_graph.ainvoke(
        {"question": "Who founded Anthropic?", "owner": "public"},
        config={"recursion_limit": 50},
    )

    assert result["route"] == "vector"
    assert result["final_answer"]
    assert result["research_report"].startswith("Anthropic was founded")
    assert result["fused_documents"]


async def test_every_node_that_ran_contributed_its_state(offline_graph):
    """The failure mode a mixed sync/async graph has and a uniform one does not: an async node
    wrapped by a sync timer returns a coroutine as its result, so the node appears to run,
    records a latency, and writes nothing. Each assertion below names a different node's
    output, so a node silently contributing nothing fails here rather than downstream."""
    result = await offline_graph.ainvoke(
        {"question": "Who founded Anthropic?", "owner": "public"},
        config={"recursion_limit": 50},
    )

    assert result["route"], "route_query contributed nothing"
    assert result["sub_queries"], "decompose_query contributed nothing"
    assert result["vector_results"], "retrieve_vector contributed nothing"
    assert result["bm25_results"], "retrieve_bm25 contributed nothing"
    assert result["fused_documents"], "fuse_results contributed nothing"
    assert result["doc_grades"], "grade_and_score contributed nothing"
    assert result["final_answer"], "synthesize_answer contributed nothing"
    assert result["research_report"], "format_report contributed nothing"


async def test_both_kinds_of_node_are_timed(offline_graph):
    """`_timed` picks its wrapper by inspecting the function, so the timing panel has to keep
    working across the sync/async boundary -- a node missing here is one whose wrapper was
    chosen wrongly."""
    result = await offline_graph.ainvoke(
        {"question": "Who founded Anthropic?", "owner": "public"},
        config={"recursion_limit": 50},
    )

    timed = {t["node"] for t in result["node_timings"]}
    assert {"route_query", "synthesize_answer"} <= timed, "an async node was not timed"
    assert {"retrieve_vector", "fuse_results", "format_report"} <= timed, (
        "a sync node was not timed"
    )
    assert all(t["latency_ms"] >= 0 for t in result["node_timings"])


async def test_a_follow_up_is_condensed_before_routing(offline_graph):
    """Condensation is the first node and rewrites what every later node reads."""
    result = await offline_graph.ainvoke(
        {
            "question": "What about their funding?",
            "owner": "public",
            "chat_history": [
                {"role": "user", "content": "Tell me about Anthropic."},
                {"role": "assistant", "content": "It is an AI safety company [1]."},
            ],
        },
        config={"recursion_limit": 50},
    )

    assert result["original_question"] == "What about their funding?"
    assert result["question"] == "Who founded Anthropic?"
