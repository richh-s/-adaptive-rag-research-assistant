from rag_assistant.graph.build_graph import _timed


def test_timed_wrapper_appends_node_timing_entry():
    def fake_node(state):
        return {"route": "vector"}

    wrapped = _timed("route_query", fake_node)
    result = wrapped({"question": "anything"})

    assert result["route"] == "vector"
    assert len(result["node_timings"]) == 1
    assert result["node_timings"][0]["node"] == "route_query"
    assert result["node_timings"][0]["latency_ms"] >= 0.0


def test_timed_wrapper_preserves_existing_result_keys():
    def fake_node(state):
        return {"vector_results": [], "bm25_results": []}

    wrapped = _timed("retrieve_vector", fake_node)
    result = wrapped({})

    assert result["vector_results"] == []
    assert result["bm25_results"] == []
    assert result["node_timings"][0]["node"] == "retrieve_vector"
