import json

from fastapi.testclient import TestClient

from rag_assistant import api


def test_health_returns_ok():
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
