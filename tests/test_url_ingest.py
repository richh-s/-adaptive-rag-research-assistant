"""Tests for POST /api/v1/ingest/url and the url_fetch helpers. Everything is offline: the
actual page fetch is stubbed at the api-module seam, and the SSRF guard is tested against
addresses that never require a network round-trip."""

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api
from rag_assistant.ingestion.build_index import IndexResult
from rag_assistant.ingestion.url_fetch import (
    FetchedPage,
    UrlIngestError,
    _assert_public_host,
    page_to_markdown,
)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    # The background task would otherwise run a real embed/index pass after the response.
    monkeypatch.setattr(
        api,
        "build_index",
        lambda on_stage=None: IndexResult(
            indexed_chunks=3, changed_files=1, skipped_files=0, removed_files=0
        ),
    )
    return TestClient(api.app)


def test_ingest_url_fetches_page_and_queues_indexing(monkeypatch, client, tmp_path):
    monkeypatch.setattr(
        api,
        "fetch_page",
        lambda url: FetchedPage(url=url, title="Example Article", text="Body text here."),
    )

    response = client.post("/api/v1/ingest/url", json={"url": "https://example.com/article"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["original_filename"] == "https://example.com/article"
    assert body["filename"].endswith(".md")

    saved = list((tmp_path / "corpus").glob("*.md"))
    assert len(saved) == 1
    content = saved[0].read_text()
    assert "# Example Article" in content
    assert "Source URL: https://example.com/article" in content
    assert "Body text here." in content


def test_ingest_url_maps_fetch_errors_to_400(monkeypatch, client):
    def _boom(url):
        raise UrlIngestError("Refusing to fetch 'localhost': it resolves to a private or local address.")

    monkeypatch.setattr(api, "fetch_page", _boom)

    response = client.post("/api/v1/ingest/url", json={"url": "http://localhost/admin"})

    assert response.status_code == 400
    assert "private or local" in response.json()["detail"]


def test_ingest_url_rejects_non_http_schemes(client):
    response = client.post("/api/v1/ingest/url", json={"url": "file:///etc/passwd"})

    assert response.status_code == 422


def test_public_host_guard_blocks_private_and_loopback_addresses():
    for url in ("http://127.0.0.1/x", "http://localhost/x", "http://192.168.1.10/x"):
        with pytest.raises(UrlIngestError):
            _assert_public_host(url)


def test_page_to_markdown_falls_back_to_netloc_when_untitled():
    page = FetchedPage(url="https://example.com/a", title=None, text="text")

    markdown = page_to_markdown(page)

    assert markdown.startswith("# example.com")
    assert "Source URL: https://example.com/a" in markdown
