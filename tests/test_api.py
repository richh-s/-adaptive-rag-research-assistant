import asyncio
import json

from fastapi.testclient import TestClient

from rag_assistant import api


def test_health_returns_ok():
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_rate_limits_per_ip(monkeypatch):
    # `_per_ip_limit`/`_global_limit` re-read settings on every request (not frozen at import
    # time), so a lowered RATE_LIMIT_RPM here takes effect without touching the limiter itself.
    # This must run before any other test hits POST /research: slowapi's in-memory storage
    # is shared for the whole process, keyed on (route, client id) not on the limit value, so
    # hits recorded under the real default limit would otherwise already count against this.
    monkeypatch.setenv("RATE_LIMIT_RPM", "1")
    monkeypatch.setattr(
        api._graph,
        "invoke",
        lambda state, config=None: {"research_report": "ok", "route": "vector", "confidence_score": 0.9},
    )
    client = TestClient(api.app)

    first = client.post("/research", json={"question": "one?"})
    second = client.post("/research", json={"question": "two?"})

    assert first.status_code == 200
    assert second.status_code == 429


def test_research_response_includes_trace_id_header(monkeypatch):
    monkeypatch.setattr(
        api._graph,
        "invoke",
        lambda state, config=None: {"research_report": "ok", "route": "vector", "confidence_score": 0.9},
    )
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "anything"})

    assert response.status_code == 200
    assert response.headers["x-trace-id"]


def test_research_passes_trace_id_into_graph_state(monkeypatch):
    captured = {}

    def _fake_invoke(state, config=None):
        captured["trace_id"] = state.get("trace_id")
        return {"research_report": "ok", "route": "vector", "confidence_score": 0.9}

    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "anything"})

    assert response.status_code == 200
    assert captured["trace_id"] == response.headers["x-trace-id"]


def test_ready_returns_200_when_both_deps_up(monkeypatch):
    monkeypatch.setattr(api, "check_chroma", lambda: (True, None))
    monkeypatch.setattr(api, "check_web_search", lambda: (True, None))
    client = TestClient(api.app)

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chroma"] == {"ok": True, "error": None}
    assert body["web_search"] == {"ok": True, "error": None}


def test_ready_returns_503_when_a_dep_is_down(monkeypatch):
    monkeypatch.setattr(api, "check_chroma", lambda: (False, "connection refused"))
    monkeypatch.setattr(api, "check_web_search", lambda: (True, None))
    client = TestClient(api.app)

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["chroma"] == {"ok": False, "error": "connection refused"}


def test_research_returns_report_and_metadata(monkeypatch):
    monkeypatch.setattr(
        api._graph,
        "invoke",
        lambda state, config=None: {
            "research_report": "The answer is 42.\n\n**Sources:**\n- [1] doc_a.md",
            "route": "vector",
            "confidence_score": 0.9,
            "node_timings": [{"node": "route_query", "latency_ms": 180.0}],
        },
    )
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "What is the answer?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is the answer?"
    assert body["report"] == "The answer is 42.\n\n**Sources:**\n- [1] doc_a.md"
    assert body["route"] == "vector"
    assert body["confidence_score"] == 0.9
    assert body["summary"]["route"] == "vector"
    assert body["summary"]["confidence_score"] == 0.9
    assert body["summary"]["total_latency_ms"] == 180.0


def test_research_rejects_empty_question():
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "   "})

    assert response.status_code == 422


def test_research_rejects_question_over_max_length():
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "a" * 2001})

    assert response.status_code == 422


def test_research_rejects_gibberish_question():
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "!!!@@@###$$$%%%^^^&&&"})

    assert response.status_code == 422


def test_research_strips_html_from_question(monkeypatch):
    captured = {}

    def _fake_invoke(state, config=None):
        captured["question"] = state["question"]
        return {"research_report": "ok", "route": "vector", "confidence_score": 0.9}

    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "<b>What is X?</b>"})

    assert response.status_code == 200
    assert captured["question"] == "What is X?"


def test_research_returns_500_on_configuration_error(monkeypatch):
    def _raise(state, config=None):
        raise RuntimeError("Missing or invalid configuration.")

    monkeypatch.setattr(api._graph, "invoke", _raise)
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "anything"})

    assert response.status_code == 500
    assert "Missing or invalid configuration." in response.json()["detail"]


def _sse_events(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def test_research_stream_yields_progress_then_done(monkeypatch):
    async def fake_astream(state, config=None, stream_mode=None):
        yield {"route_query": {"route": "vector"}}
        yield {"synthesize_answer": {"final_answer": "The answer is 42."}}
        yield {
            "format_report": {
                "research_report": "The answer is 42.\n\n**Sources:**\n- [1] doc_a.md",
                "route": "vector",
                "confidence_score": 0.9,
                "node_timings": [{"node": "route_query", "latency_ms": 180.0}],
            }
        }

    monkeypatch.setattr(api._graph, "astream", fake_astream)
    client = TestClient(api.app)

    with client.stream(
        "POST", "/research/stream", json={"question": "What is the answer?"}
    ) as response:
        assert response.status_code == 200
        events = _sse_events(response)

    assert events[0]["type"] == "progress"
    assert events[0]["node"] == "route_query"
    assert events[-1]["type"] == "done"
    assert events[-1]["report"] == "The answer is 42.\n\n**Sources:**\n- [1] doc_a.md"
    assert events[-1]["route"] == "vector"
    assert events[-1]["confidence_score"] == 0.9
    assert events[-1]["summary"]["total_latency_ms"] == 180.0


def test_research_stream_accumulates_send_fanned_fields_across_updates(monkeypatch):
    async def fake_astream(state, config=None, stream_mode=None):
        yield {"route_query": {"route": "vector", "node_timings": [{"node": "route_query", "latency_ms": 100.0}]}}
        yield {"retrieve_vector": {"vector_results": [], "node_timings": [{"node": "retrieve_vector", "latency_ms": 50.0}]}}
        yield {"retrieve_vector": {"vector_results": [], "node_timings": [{"node": "retrieve_vector", "latency_ms": 60.0}]}}
        yield {
            "format_report": {
                "research_report": "The answer is 42.",
                "confidence_score": 0.9,
                "node_timings": [{"node": "format_report", "latency_ms": 10.0}],
            }
        }

    monkeypatch.setattr(api._graph, "astream", fake_astream)
    client = TestClient(api.app)

    with client.stream(
        "POST", "/research/stream", json={"question": "What is the answer?"}
    ) as response:
        assert response.status_code == 200
        events = _sse_events(response)

    latencies = events[-1]["summary"]["node_latencies_ms"]
    assert len(latencies) == 4
    assert events[-1]["summary"]["total_latency_ms"] == 220.0


def test_research_stream_times_out_on_hanging_node(monkeypatch):
    monkeypatch.setenv("GRAPH_TIMEOUT_SECONDS", "0.05")

    async def fake_astream(state, config=None, stream_mode=None):
        yield {"route_query": {"route": "vector"}}
        await asyncio.sleep(10)
        yield {"synthesize_answer": {"final_answer": "unreachable"}}

    monkeypatch.setattr(api._graph, "astream", fake_astream)
    client = TestClient(api.app)

    with client.stream(
        "POST", "/research/stream", json={"question": "anything"}
    ) as response:
        assert response.status_code == 200
        events = _sse_events(response)

    assert events[0]["type"] == "progress"
    assert events[-1]["type"] == "error"
    assert "timed out" in events[-1]["detail"]


def test_sigterm_handler_sets_shutdown_event():
    assert not api._shutdown_event.is_set()
    try:
        api._handle_sigterm()
        assert api._shutdown_event.is_set()
    finally:
        api._shutdown_event.clear()


def test_research_stream_sends_close_event_on_shutdown(monkeypatch):
    async def fake_astream(state, config=None, stream_mode=None):
        yield {"route_query": {"route": "vector"}}
        yield {"synthesize_answer": {"final_answer": "unreachable"}}

    monkeypatch.setattr(api._graph, "astream", fake_astream)
    api._shutdown_event.set()
    try:
        client = TestClient(api.app)

        with client.stream(
            "POST", "/research/stream", json={"question": "anything"}
        ) as response:
            assert response.status_code == 200
            events = _sse_events(response)

        assert events[-1]["type"] == "close"
    finally:
        api._shutdown_event.clear()


def test_research_stream_yields_error_event_on_failure(monkeypatch):
    async def fake_astream(state, config=None, stream_mode=None):
        yield {"route_query": {"route": "vector"}}
        raise RuntimeError("Missing or invalid configuration.")

    monkeypatch.setattr(api._graph, "astream", fake_astream)
    client = TestClient(api.app)

    with client.stream(
        "POST", "/research/stream", json={"question": "anything"}
    ) as response:
        assert response.status_code == 200
        events = _sse_events(response)

    assert events[0]["type"] == "progress"
    assert events[-1]["type"] == "error"
    assert "Missing or invalid configuration." in events[-1]["detail"]
