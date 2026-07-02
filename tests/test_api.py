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


def test_research_returns_500_on_configuration_error(monkeypatch):
    def _raise(state, config=None):
        raise RuntimeError("Missing or invalid configuration.")

    monkeypatch.setattr(api._graph, "invoke", _raise)
    client = TestClient(api.app)

    response = client.post("/research", json={"question": "anything"})

    assert response.status_code == 500
    assert "Missing or invalid configuration." in response.json()["detail"]
