from rag_assistant.graph.research_summary import build_research_summary
from rag_assistant.schemas.models import FusedDocument, RetrievedDoc, SubQueryResult


def test_build_research_summary_counts_docs_per_source():
    state = {
        "route": "both",
        "sub_queries": ["q1", "q2"],
        "vector_results": [
            SubQueryResult(sub_query="q1", docs=[RetrievedDoc(content="a", source_id="a.md")]),
            SubQueryResult(sub_query="q2", docs=[RetrievedDoc(content="b", source_id="b.md")]),
        ],
        "bm25_results": [
            SubQueryResult(sub_query="q1", docs=[RetrievedDoc(content="a", source_id="a.md")]),
        ],
        "web_results": [
            SubQueryResult(
                sub_query="q1",
                docs=[
                    RetrievedDoc(content="c", source_id="c.md"),
                    RetrievedDoc(content="d", source_id="d.md"),
                ],
            ),
        ],
        "fused_documents": [
            FusedDocument(content="a", source_id="a.md", rrf_score=0.9),
            FusedDocument(content="b", source_id="b.md", rrf_score=0.5),
        ],
        "confidence_score": 0.91,
        "correction_attempted": False,
        "node_timings": [
            {"node": "route_query", "latency_ms": 180.0},
            {"node": "retrieve_vector", "latency_ms": 300.0},
            {"node": "retrieve_vector", "latency_ms": 250.0},
        ],
    }

    summary = build_research_summary(state)

    assert summary.route == "both"
    assert summary.sub_queries == ["q1", "q2"]
    assert summary.retrieval_counts.vector == 2
    assert summary.retrieval_counts.bm25 == 1
    assert summary.retrieval_counts.web == 2
    assert summary.fused_document_count == 2
    assert summary.confidence_score == 0.91
    assert summary.correction_attempted is False
    assert len(summary.node_latencies_ms) == 3
    assert summary.total_latency_ms == 730.0


def test_build_research_summary_handles_missing_state_keys():
    summary = build_research_summary({})

    assert summary.route is None
    assert summary.sub_queries == []
    assert summary.retrieval_counts.vector == 0
    assert summary.fused_document_count == 0
    assert summary.confidence_score is None
    assert summary.correction_attempted is False
    assert summary.node_latencies_ms == []
    assert summary.total_latency_ms == 0.0
