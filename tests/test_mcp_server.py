"""Tests for the MCP server's tools. FastMCP's @tool() decorator registers and returns the
original function, so each tool is tested as a plain function with the pipeline mocked --
the protocol/stdio layer is the SDK's responsibility, not ours."""

from rag_assistant import mcp_server
from rag_assistant.ingestion.build_index import IndexResult
from rag_assistant.ingestion.url_fetch import FetchedPage, UrlIngestError


class _FakeGraph:
    def __init__(self, state):
        self.state = state
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append(state)
        return self.state


def _index_result():
    return IndexResult(indexed_chunks=4, changed_files=1, skipped_files=0, removed_files=0)


def test_research_question_returns_report_with_transparency_footer(monkeypatch):
    fake = _FakeGraph(
        {
            "research_report": "# Answer\nAnthropic was founded in 2021. [1]",
            "route": "vector",
            "confidence_score": 0.9,
            "fused_documents": [],
        }
    )
    monkeypatch.setattr(mcp_server, "_get_graph", lambda: fake)

    output = mcp_server.research_question("Who founded Anthropic?")

    assert output.startswith("# Answer")
    assert "route: vector" in output
    assert "confidence: 0.9" in output
    assert fake.calls[0]["question"] == "Who founded Anthropic?"


def test_ingest_file_copies_into_corpus_and_indexes(monkeypatch, tmp_path):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setattr("rag_assistant.ingestion.build_index.build_index", lambda: _index_result())
    source = tmp_path / "notes.md"
    source.write_text("# Meeting notes")

    output = mcp_server.ingest_file(str(source))

    assert "Ingested notes.md" in output
    assert (tmp_path / "corpus" / "notes.md").read_text() == "# Meeting notes"


def test_ingest_file_rejects_unsupported_and_missing(monkeypatch, tmp_path):
    bad = tmp_path / "data.xyz"
    bad.write_text("x")

    assert "unsupported file type" in mcp_server.ingest_file(str(bad))
    assert "is not a file" in mcp_server.ingest_file(str(tmp_path / "nope.md"))


def test_ingest_url_writes_page_and_indexes(monkeypatch, tmp_path):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setattr("rag_assistant.ingestion.build_index.build_index", lambda: _index_result())
    monkeypatch.setattr(
        "rag_assistant.ingestion.url_fetch.fetch_page",
        lambda url: FetchedPage(url=url, title="An Article", text="Body."),
    )

    output = mcp_server.ingest_url("https://example.com/a")

    assert "Ingested 'An Article'" in output
    saved = list((tmp_path / "corpus").glob("*.md"))
    assert len(saved) == 1
    assert "Source URL: https://example.com/a" in saved[0].read_text()


def test_ingest_url_surfaces_fetch_errors_as_text(monkeypatch):
    def _blocked(url):
        raise UrlIngestError("Refusing to fetch 'localhost'.")

    monkeypatch.setattr("rag_assistant.ingestion.url_fetch.fetch_page", _blocked)

    assert "Refusing to fetch" in mcp_server.ingest_url("http://localhost/x")
    assert "must start with http" in mcp_server.ingest_url("ftp://example.com")


def test_list_documents_empty_knowledge_base(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

    assert "empty" in mcp_server.list_documents()


def test_list_conversations_lists_saved_rows(monkeypatch):
    from rag_assistant.conversations import store

    conversation = store.create_conversation("About Anthropic")
    store.append_turn(conversation.id, question="q", answer="a")

    output = mcp_server.list_conversations()

    assert "About Anthropic" in output
    assert conversation.id in output
