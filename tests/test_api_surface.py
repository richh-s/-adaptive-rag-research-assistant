"""Tests for the deployment-shaped parts of the HTTP surface: route versioning and CORS.

Both are things that only bite in a configuration other than the developer's own -- a client
pinned to the pre-versioning path, or a frontend served from a different origin than the API.
"""

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api
from rag_assistant.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """slowapi's hit counts are process-wide and other test modules consume the default
    budget; these tests are about routing, not throttling."""
    monkeypatch.setenv("RATE_LIMIT_RPM", "10000")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "10000")


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


# ---- route versioning ----


@pytest.mark.parametrize("path", ["/api/v1/research", "/research"])
def test_both_versioned_and_legacy_research_paths_are_served(client: TestClient, monkeypatch, path):
    """The unversioned path stays registered so clients written before versioning keep
    working; a 404 here would be a silent breaking change for them."""
    monkeypatch.setattr(
        api._graph,
        "invoke",
        lambda *args, **kwargs: {
            "research_report": "# Report",
            "final_answer": "An answer.",
            "route": "vector",
            "confidence_score": 0.9,
        },
    )

    response = client.post(path, json={"question": "Who founded Anthropic?", "save": False})

    assert response.status_code == 200
    assert response.json()["answer"] == "An answer."


@pytest.mark.parametrize("path", ["/api/v1/research/stream", "/research/stream"])
def test_both_versioned_and_legacy_stream_paths_are_served(client: TestClient, monkeypatch, path):
    async def _fake_astream(*args, **kwargs):
        yield {"format_report": {"research_report": "# Report", "final_answer": "Answer."}}

    monkeypatch.setattr(api._graph, "astream", _fake_astream)

    response = client.post(path, json={"question": "Who founded Anthropic?", "save": False})

    assert response.status_code == 200
    assert '"type":"done"' in response.text


def test_only_the_versioned_research_paths_appear_in_the_openapi_schema(client: TestClient):
    """One obvious path in the docs; the legacy aliases are compatibility shims, not API."""
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/research" in paths
    assert "/api/v1/research/stream" in paths
    assert "/research" not in paths
    assert "/research/stream" not in paths


def test_legacy_research_paths_are_still_rate_limited(client: TestClient, monkeypatch):
    """The alias must not be a way around the limiter -- otherwise documenting the versioned
    path while leaving the old one uncapped would be a hole rather than a courtesy."""
    monkeypatch.setenv("RATE_LIMIT_RPM", "1")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "10000")
    # slowapi's in-memory counters are process-wide and every test in the run shares the
    # "testclient" caller identity, so earlier modules have already spent part of this
    # minute's budget -- without a reset the *first* request here would 429 and the test
    # would pass or fail depending on file ordering.
    api.limiter.reset()
    api.global_limiter.reset()
    monkeypatch.setattr(
        api._graph,
        "invoke",
        lambda *args, **kwargs: {"research_report": "# Report", "route": "none"},
    )

    first = client.post("/research", json={"question": "A question here.", "save": False})
    second = client.post("/research", json={"question": "A question here.", "save": False})

    assert first.status_code == 200
    assert second.status_code == 429


# ---- CORS ----


def test_cors_origins_default_to_the_vite_dev_server(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()

    assert "http://localhost:5173" in get_settings().cors_origins()


def test_cors_origins_are_parsed_from_a_comma_separated_setting():
    settings = Settings(
        google_api_key="k",
        anthropic_api_key="",
        cors_allow_origins="https://app.example.com, https://staging.example.com ",
    )

    assert settings.cors_origins() == ["https://app.example.com", "https://staging.example.com"]


def test_blank_cors_setting_means_no_cross_origin_access():
    """Correct for the single-container deploy, where the frontend is same-origin and any
    allowed origin would be strictly more permission than the app needs."""
    settings = Settings(google_api_key="k", anthropic_api_key="", cors_allow_origins="")

    assert settings.cors_origins() == []


def test_preflight_from_an_allowed_origin_is_accepted(client: TestClient):
    response = client.options(
        "/api/v1/research",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_preflight_from_an_unlisted_origin_is_not_granted(client: TestClient):
    response = client.options(
        "/api/v1/research",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_trace_id_header_is_exposed_to_the_browser(client: TestClient):
    """Without `expose_headers` a cross-origin frontend can read the response but not
    X-Trace-Id -- which makes the trace ID useless for exactly the person reporting the bug."""
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert "X-Trace-Id" in response.headers["access-control-expose-headers"]
