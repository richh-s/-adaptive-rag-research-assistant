from rag_assistant.graph.nodes.fuse import fuse_results
from rag_assistant.schemas.models import RetrievedDoc, SubQueryResult


def test_fuse_results_merges_and_ranks_across_subqueries_and_sources():
    state = {
        "vector_results": [
            SubQueryResult(
                sub_query="q1", docs=[RetrievedDoc(content="doc a", source_id="a.md")]
            ),
            SubQueryResult(
                sub_query="q2", docs=[RetrievedDoc(content="doc b", source_id="b.md")]
            ),
        ],
        "web_results": [
            SubQueryResult(
                sub_query="q1", docs=[RetrievedDoc(content="doc a", source_id="a.md")]
            ),
        ],
    }

    fused = fuse_results(state)["fused_documents"]

    assert len(fused) == 2
    assert fused[0].content == "doc a"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_fuse_results_handles_missing_state_keys():
    assert fuse_results({})["fused_documents"] == []
